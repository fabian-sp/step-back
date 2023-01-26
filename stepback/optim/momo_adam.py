"""
Some parts of the code are adapted from:
    Defazio, Aaron: https://github.com/facebookresearch/dadaptation/blob/main/dadaptation/dadapt_adam.py.
"""
import math
import warnings
import torch
import torch.optim
from ..types import Params, LossClosure, OptFloat


class MomoAdam(torch.optim.Optimizer):
    r"""
    """
    def __init__(self, 
                params: Params, 
                lr: float=1.0, 
                betas:tuple=(0.9, 0.999), 
                eps:float=1e-8,
                weight_decay:float=0):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay
                        )
        
        super().__init__(params, defaults)

        self._number_steps = 0
        self.state['step_size_list'] = list() # for storing

        return

    def step(self, closure: LossClosure=None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of first group.")
            warnings.warn("More than one param group. Check if step method is correct for this case.")

        group = self.param_groups[0]
        beta1, beta2 = group['betas']
        
        _dot = 0. # = <d_k,x_k>
        _gamma = 0. # = gamma_l
        _norm = 0 # = ||d_k||^2_{D_k^-1}

        # Exponential moving average of function value
        if self._number_steps >= 1:
            self.loss_avg = (1-beta1)*loss +  beta1*self.loss_avg  
        else:
            self.loss_avg = loss # initialize

        for group in self.param_groups:
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data                
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradients
                    state['grad_avg'] = grad.detach()
                    # Exponential moving average of squared gradient values
                    state['grad_avg_sq'] = torch.mul(grad, grad).sqrt().detach()
                    # Exponential moving average of inner product <grad, weight>
                    state['grad_dot_w'] = torch.sum(torch.mul(p.data, grad))
                    
                grad_avg, grad_avg_sq = state['grad_avg'], state['grad_avg_sq']
                grad_dot_w = state['grad_dot_w']

                # Adam EMA updates
                grad_avg.mul_(beta1).add_(grad, alpha=1-beta1) # = d_k
                grad_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) # = v_k
                grad_dot_w.mul_(beta1).add_(torch.sum(torch.mul(p.data, grad)), alpha=1-beta1)

                Dk = grad_avg_sq.sqrt().add_(eps) # = D_k
                
                _dot.add_(torch.sum(torch.mul(p.data, grad_avg)))
                _gamma.add_(grad_dot_w)
                _norm.add_(grad_avg.mul(grad_avg.div(Dk)))

        #################   
        # Update
        for group in self.param_groups:
            
            lr = group['lr']
            decay = group['weight_decay']
            eps = group['eps']

            nom = max(self.loss_avg - self.lb + _dot - _gamma, 0)
            denom = _norm
            t1 = (nom/denom).item()
            tau = min(lr, t1)

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                grad_avg, grad_avg_sq = state['grad_avg'], state['grad_avg_sq']

                Dk = grad_avg_sq.sqrt().add_(eps) # = D_k
                p.data.addcdiv_(grad_avg, Dk, value=-tau) # x_k - tau*(d_k/D_k)
                state['step'] += 1

        #############################
        ## Maintenance

        self._number_steps += 1
        self.state['step_size_list'].append(t1)

        return loss