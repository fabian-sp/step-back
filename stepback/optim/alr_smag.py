"""
Implementation of the ALR-SMAG algorithm.
https://arxiv.org/pdf/2305.12939.pdf

Author (of the code): Fabian Schaipp
"""
import torch
import warnings
from math import sqrt

from ..types import Params, LossClosure, OptFloat

class AlrSmag(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-1,
                 weight_decay: float=0,
                 beta: float=0.9,
                 lb: float=0,
                 c: float=1) -> None:
        """
        ALR-SMAG optimizer

        Parameters
        ----------
        params : Params
            Model parameters.
        lr : float, optional
            Learning rate, by default 1e-1.
        weight_decay : float, optional
            Weight decay parameter, by default 0.
        beta : float, optional
            Momentum parameter, should be in [0,1), by default 0.9.
        lb : float, optional
            Lower bound for loss. By default 0.
        c : Paramater c in the paper, by default 1.
        """
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if (beta < 0.0) or (beta > 1.0):
            raise ValueError("Invalid beta parameter: {}".format(beta))
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(AlrSmag, self).__init__(params, defaults)
        
        self.beta = beta
        self.lb = lb
        self.c = c

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
        beta = self.beta  
        
        ###### Preliminaries
        _norm = 0.
        
        ############################################################
        # Notation
        # d_k: p.grad_avg
        # Note that in this paper it actually is not an average
        for group in self.param_groups:
            for p in group['params']:
                
                grad = p.grad.data.detach()
                state = self.state[p]

                # Initialize d_k
                if self._number_steps == 1:
                    state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                        
                grad_avg = state['grad_avg']

                grad_avg.mul_(beta).add_(grad, alpha=1)                
                _norm += torch.sum(torch.mul(grad_avg, grad_avg))

        #################   
        # Update
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            
            ### Compute adaptive step size
            t1 = (loss.detach() - self.lb)/(self.c * _norm)
            t1 = t1.item() # make scalar
            
            tau = min(lr, t1) # step size

            ### Update params
            for p in group['params']:   
                state = self.state[p]
                grad_avg = state['grad_avg']          
                
                if lmbda > 0:
                    p.data.mul_(1-lmbda*tau)

                p.data.add_(other=grad_avg, alpha=-tau)
                    
        ############################################################    
        self.state['step_size_list'].append(t1)
        
        return loss
