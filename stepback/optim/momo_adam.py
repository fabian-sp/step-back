"""
Some parts of the code are adapted from https://github.com/facebookresearch/dadaptation/blob/main/dadaptation/dadapt_adam.py.
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

        group = self.param_groups[0]
        beta1, beta2 = group['betas']
        norm = 0 # = ||d_k||^2_{D_k^-1}

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
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.mul(grad, grad).sqrt().detach()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                

                # Adam EMA updates
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1) # = d_k
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) # = v_k
                
                Dk = exp_avg_sq.sqrt().add_(eps) # = D_k
                norm.add_(exp_avg.mul((1/Dk).mul(exp_avg)))


            ######

        # FROM HERE ON FINALIZE
        for group in self.param_groups:
            
            decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                Dk = exp_avg_sq.sqrt().add_(eps) # = D_k

                
                ### Take step
                #p.data.addcdiv_(exp_avg, denom, value=-1)
                
                state['step'] += 1


        return loss