"""
Implements the Lion algorithm.

See: https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py
"""
import torch
import warnings
from math import sqrt

from ..types import Params, LossClosure, OptFloat

class Lion(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-4,
                 weight_decay: float=0,
                 betas: tuple=(0.9,0.99),
                 ) -> None:
        """
        Lion optimizer

        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate, by default 1e-4.
        weight_decay : float, optional
            Weight decay parameter, by default 0.
        betas : tuple, optional
            Momentum parameters, should be in [0,1), by default (0.9,0.99).
        """
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        for beta in betas:
            if (beta < 0.0) or (beta > 1.0):
                raise ValueError("Invalid beta parameter: {}".format(beta))
            
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas)
        
        super(Lion, self).__init__(params, defaults)
        
        return
        
    def step(self, closure: LossClosure=None) -> OptFloat:
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss, by default None.
        
        Returns
        -------
        (Stochastic) Loss function value.
        """
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        ############################################################
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            beta1, beta2 = group['betas']
  
            for p in group['params']:
                
                grad = p.grad.data.detach()
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradients
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    
                state['step'] += 1       
                exp_avg = state['exp_avg']
                
                # Update
                if lmbda > 0:
                    p.data.mul_(1 - lr * lmbda)
                
                vk = exp_avg.clone().mul_(beta1).add(grad, alpha=1-beta1).sign_()
                # update params
                p.data.add_(vk, alpha=-lr)
                
                # update dk
                exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)
            
        return loss