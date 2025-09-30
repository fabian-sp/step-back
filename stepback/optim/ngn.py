"""
Implements the NGN algorithm by Orvieto and Xiao.

Reference: https://arxiv.org/pdf/2407.04358
"""
import torch
import warnings
from math import sqrt

from ..types import Params, LossClosure, OptFloat

class NGN(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-1,
        ) -> None:
        """
        NGN optimizer

        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate, by default 1e-1.
        """
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        defaults = dict(lr=lr)
        
        super(NGN, self).__init__(params, defaults)
        
        # Initialization
        self._number_steps = 0
        self.state['step_size_list'] = list() # for storing the adaptive step size term
        
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
        assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
        assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")

        self._number_steps += 1
        
        # Update
        grad_norm = self.compute_grad_norm()
        for group in self.param_groups:
            lr = group['lr']
            denom = 1 + lr / (2*loss) * (grad_norm**2)
            gamma = (lr / denom).item()
            
            ### Update params
            for p in group['params']:
                p.data.add_(other=p.grad.data, alpha=-gamma)
            
        self.state['step_size_list'].append(gamma)
        
        return loss
    
    @torch.no_grad()
    def compute_grad_norm(self):
        grad_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                assert p.grad is not None
                
                g = p.grad.data
                grad_norm += torch.sum(torch.mul(g, g))
                
        grad_norm = torch.sqrt(grad_norm)
        return grad_norm
