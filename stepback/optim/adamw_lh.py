"""
Adapted from https://github.com/zhenxun-zhuang/AdamW-Scale-free/blob/main/src/adam.py#L67

The main difference to the Pytorch AdamW implementation is that we use the
weight decay coefficient 
    (1-inital_lr*weight_decay) 
instead of 
    (1-lr*weight_decay) 

This is like originally proposed in the Loshchilov, Hutter paper (thus the name AdamLH).

Decoupled Weight Decay Regularization:
    https://arxiv.org/abs/1711.05101
"""

import math
import torch
from torch.optim import Optimizer


class AdamLH(Optimizer):
    """ AdamW with fully decoupled weight decay.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        self._init_lr = lr
        super(AdamLH, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamLH, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
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
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # decay
                p.mul_(1 - lmbda*lr/self._init_lr)

                grad = p.grad
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']


                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1
                update = -step_size * exp_avg / denom
                p.add_(update)
                
        return loss