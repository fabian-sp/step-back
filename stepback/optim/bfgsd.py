"""
Implements the BFGS lambda diagonal method
"""
import math
import warnings
import torch
import torch.optim
from ..types import Params, LossClosure, OptFloat

class BFGSd(torch.optim.Optimizer):
    def __init__(self, 
                params: Params, 
                lr: float=1.0, 
                betas:tuple=(0.9, 0.999), 
                lmbda:float=10.0,
                weight_decay:float=0):
        """
        BFGS lambda diagonal optimizer

        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate, by default 1.0.
        betas : tuple, optional
            Momentum parameters for running avergaes and its square. By default (0.9, 0.999).
        lmbda : float, optional
            regularization parameter of KL term
        weight_decay : float, optional
            Weight decay parameter, by default 0.
        divide : bool, optional
            Whether to do proximal update (divide=True) or the AdamW approximation (divide=False), by default True.

        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lmbda:
            raise ValueError("Invalid epsilon value: {}".format(lmbda))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        
        defaults = dict(lr=lr, betas=betas, lmbda=lmbda,
                        weight_decay=weight_decay
                        )
        
        super().__init__(params, defaults)

        self.lmbda = lmbda        
        return

    def step(self, closure: LossClosure=None, loss: torch.Tensor=None) -> OptFloat:
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss, by default None.

        loss : torch.tensor, optional
            The loss tensor. Use this when the backward step has already been performed. By default None.
            
        Returns
        -------
        (Stochastic) Loss function value.
        """
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")

        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
        
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data           
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Diagonal quasi-Newton matrix
                    state['h'] = torch.ones_like(p.data, memory_format=torch.preserve_format).detach()
                    # Difference of iterates
                    state['s'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Difference of grads
                    state['y'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Momentum buffer of gradients
                    state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                state['step'] += 1 

                grad_avg = state['grad_avg']
                grad_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                bias_correction1 = 1 - beta1 ** state['step']
                grad_avg.div_(bias_correction1)
        #################   
        # update iterate
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            beta1, beta2 = group['betas']
            
            ### Update params
            for p in group['params']:
                if p.grad is None:
                    continue   
                grad = p.grad.data
                state = self.state[p]
                grad_avg = state['grad_avg']
                h = state['h']
                s = state['s']
                y = state['y']
                y.sub_(grad)
                s.sub_(p.data)
                # AdamW way of adding weight decay
                if wd > 0:
                    p.data.mul_(1-wd*lr)
                # Gradient step  x =x -lr*h*grad
                import pdb; pdb.set_trace()
                p.data.addcmul_(grad_avg, h, value=-lr) # x_k - tau*(d_k/D_k)

        import pdb; pdb.set_trace()
        #Recompute the gradient at the new iterate, but same batch
        with torch.enable_grad(): 
            loss = closure()

        #############################
        ## Update quasi-Newton matrix
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            beta1, beta2 = group['betas']
            lmbda = group['lmbda']
            
            ### Update params
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                h = state['h']
                s = state['s']
                y = state['y']
                y.add_(grad)
                s.add_(p.data)
                # h = (sqrt{1+ 4*lmbda  y^2(h+lmbda *s^2)}-1)/(2*lmbda*y^2)
                s.square_()
                y.square_()
                h.add_(s.mul_(lmbda)).mul_(y).mul_(4*lmbda).add_(1.0).sqrt_().sub_(-1.0).div_(y.mul_(2.0*lmbda))  
                # reset for next iteration
                s.zero_()
                y.zero_()

        return loss