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
                betas:tuple=(1.0, 0.0), 
                lmbda:float=10.0,
                weight_decay:float=0,
                eps = 1e-10,
                clampmin = 0.01,
                clampmax = 100.0
                ):
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
        eps : float, optional
            Added to denominator of Hessian update to avoid dividing by almost zero
        clampmin, clampmax: float, option
            Clamps the hess between these two values.

        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lmbda:
            raise ValueError("Invalid lmbda value: {}".format(lmbda))
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        
        defaults = dict(lr=lr, betas=betas, lmbda=lmbda,
                        weight_decay=weight_decay
                        )
        
        super().__init__(params, defaults)

        self.lmbda = lmbda  
        self.eps = eps      
        self.clampmin = clampmin
        self.clampmax =  clampmax
        self.state['step_size_list'] = list() 
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

        #################   
        # update iterate and momentum buffer
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data           
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Diagonal quasi-Newton matrix
                    state['hess'] = torch.ones_like(p.data, memory_format=torch.preserve_format).detach()
                    # Difference of iterates
                    state['xdiff'] = -p.data.clone()
                    # Difference of grads
                    state['graddiff'] = -grad.clone()
                    # state['xdiff'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # state['graddiff'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # # Momentum buffer of gradients
                    state['grad_avg'] = grad.clone()
                state['step'] += 1 
                grad_avg = state['grad_avg']
                grad_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                hess =state['hess']
                xdiff = state['xdiff']
                graddiff = state['graddiff']
                graddiff.mul_(beta2).sub_(grad, alpha=1-beta2)
                xdiff.mul_(beta2).sub_(p.data, alpha=1-beta2)
                # xdiff.sub_(p.data)
                # AdamW way of adding weight decay
                if wd > 0:
                    p.data.mul_(1-wd*lr)
                # Gradient step  x =x -lr*h*grad
                p.data.addcmul_(grad_avg, hess, value=-lr) # x_k - tau*(d_k/D_k)
                p.grad.data.zero_()
    
        #Recompute the gradient at the new iterate, but same batch
        with torch.enable_grad(): 
            loss = closure()

        #############################
        ## Update quasi-Newton matrix
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            lmbda = group['lmbda']
            
            ### Update params
            num_groups =1
            hess_mean =0
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                hess =state['hess']
                xdiff = state['xdiff']
                graddiff = state['graddiff']
                # bias_correction2 = 1 - beta2 ** state['step']
                graddiff.add_(grad, alpha=1-beta2) #.div_(bias_correction2)
                xdiff.add_(p.data, alpha=1-beta2) #.div_(bias_correction2)
                # graddiff.add_(grad)
                # xdiff.add_(p.data)
                # xdiff.square_()
                # graddiff.square_()
                #hess =(sqrt{1+ 4*lmbda  y^2(h+lmbda *s^2)}-1)/(2*lmbda*y^2)
                # hess3 = ((1.0+ 4.0*lmbda *graddiff*(hess + lmbda*xdiff)).sqrt()-1.0).add(self.eps).div(2.0*lmbda*graddiff+self.eps)
                # hess2 = hess.add(xdiff.mul(lmbda)).mul(graddiff).mul(4.0*lmbda).add(1.0).sqrt().sub(1.0).div(graddiff.mul(2.0*lmbda))  
                # In numpy: 
                # H[:] =  H + lmbda*(s**2)
                # H[:] =  (np.sqrt(1+ 4*lmbda*(y**2)*(H))-1+eps)/(2*lmbda* (y**2)+eps) 
                hess.add_(xdiff.square(), alpha =lmbda)
                hess.mul_(graddiff.square()).mul_(4.0*lmbda).add_(1.0).sqrt_().sub_(1.0).add_(self.eps).div_(graddiff.square().mul(2.0*lmbda).add(self.eps))  
                # reset for next iteration
                # print(hess.mean())
                # print(hess3.mean())
                hess_mean = (hess_mean*(num_groups-1)+hess.mean().item())/num_groups
                num_groups = num_groups+1
                # xdiff.zero_()
                # graddiff.zero_()
                hess.clamp_(self.clampmin, self.clampmax)
                # print(hess)

        self.state['step_size_list'].append(hess_mean)
        return loss