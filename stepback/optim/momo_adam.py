"""
Some parts of the code are adapted from:
    Defazio, Aaron: https://github.com/facebookresearch/dadaptation/blob/main/dadaptation/dadapt_adam.py.
"""
import math
import warnings
import torch
import torch.optim
from ..types import Params, LossClosure, OptFloat
import numpy as np

class MomoAdam(torch.optim.Optimizer):
    r"""
    Implements Adam with adaptive step sizes.
    """
    def __init__(self, 
                params: Params, 
                lr: float=1.0, 
                betas:tuple=(0.9, 0.999), 
                eps:float=1e-8,
                weight_decay:float=0,
                lb: float=0,
                divide: bool=True,
                use_fstar: bool=False):

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

        self.lb = lb
        self.divide = divide 
        self.eta = 1.0
        self.omega = 0.0
        self.use_fstar = use_fstar
        
        # initialize
        self._number_steps = 0
        self.loss_avg = 0.
        self.state['step_size_list'] = list() # for storing

        return

    def step(self, closure: LossClosure=None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        with torch.enable_grad():
            loss = closure()
        
        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")

        _dot = 0. # = <d_k,x_k>
        _gamma = 0. 
        _grad_norm = 0. # = ||d_k||^2_{D_k^-1}
        _delta = 1.

        self._number_steps += 1
        _use_fstar_this_iter = self.use_fstar and (self._number_steps >= 2)

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
  
            bias_correction1 = 1 - beta1 ** self._number_steps
            bias_correction2 = 1 - beta2 ** self._number_steps
        
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data           
                state = self.state[p]

                # OLD: NO BIAS CORRECTION
                # State initialization
                # if 'step' not in state:
                #    state['step'] = 0
                #    # Exponential moving average of gradients
                #    state['grad_avg'] = grad.clone().detach()
                #    # Exponential moving average of squared gradient values
                #    state['grad_avg_sq'] = torch.mul(grad, grad).detach()
                #    # Exponential moving average of inner product <grad, weight>
                #    state['grad_dot_w'] = torch.sum(torch.mul(p.data, grad))

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradients
                    state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of squared gradient values
                    state['grad_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of inner product <grad, weight>
                    state['grad_dot_w'] = torch.tensor(0.).to(p.device)
                

                state['step'] += 1 # increment iteration counter
                grad_avg, grad_avg_sq = state['grad_avg'], state['grad_avg_sq']
                grad_dot_w = state['grad_dot_w']

                if _use_fstar_this_iter:
                    # compute before updating EMA
                    Dk_minus_1 = grad_avg_sq.div(1 - beta2**(self._number_steps-1)).sqrt().add(eps) 

                # Adam EMA updates
                grad_avg.mul_(beta1).add_(grad, alpha=1-beta1) # = d_k
                grad_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2) # = v_k
                grad_dot_w.mul_(beta1).add_(torch.sum(torch.mul(p.data, grad)), alpha=1-beta1)

                bias_correction2 = 1 - beta2 ** state['step']
                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps) # = D_k
                
                with torch.no_grad():
                    _dot += torch.sum(torch.mul(p.data, grad_avg))
                    _gamma += grad_dot_w
                    _grad_norm += torch.sum(grad_avg.mul(grad_avg.div(Dk)))
                    
                    if _use_fstar_this_iter:
                        ratio = Dk_minus_1/Dk
                        _delta_this_param = torch.min(ratio).item()                
                        _delta = min(_delta_this_param, _delta)
        
        if _use_fstar_this_iter:
            self.eta *= _delta

        # Exponential moving average of function value
        # Uses beta1 of last param_group! 
        self.loss_avg = (1-beta1)*loss +  beta1*self.loss_avg 
        
        # OLD: NO BIAS CORRECTION
        # if self._number_steps >= 1:
        #     self.loss_avg = (1-beta1)*loss +  beta1*self.loss_avg  
        # else:
        #     self.loss_avg = loss.clone().detach() # initialize

        #################   
        # Update
        for group in self.param_groups:
            
            ### Compute adaptive step size
            lr = group['lr']
            lmbda = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']

            bias_correction1 = 1 - beta1 ** self._number_steps
            bias_correction2 = 1 - beta2 ** self._number_steps

            if _use_fstar_this_iter:  
                h = (self.loss_avg  +  _dot - _gamma).item()              
                # RESET
                if (1-1./np.sqrt(self._number_steps))*h < bias_correction1*self.lb:
                    self.lb = 0. 
                    self.eta = 1.
                    self.omega = 0.
                    
            nom = (1+lr*lmbda) * (self.loss_avg - bias_correction1*self.lb)  + _dot - (1+lr*lmbda) * _gamma
                
            t1 = (max(nom, 0.)/_grad_norm).item()
            tau = min(lr/bias_correction1, t1)
            
            ### Update lb estimator
            if _use_fstar_this_iter:
                omega_tmp = self.omega # omega_k
                self.omega += self.eta * tau * bias_correction1 # omega_{k+1}
                self.lb = ((self.lb*omega_tmp + self.eta*tau*(2*h - tau*_grad_norm)) / self.omega).item()

            ### Update params
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                grad_avg, grad_avg_sq = state['grad_avg'], state['grad_avg_sq']

                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps)
                
                # AdamW-Pytorch way of weight decay
                if lmbda > 0 and not self.divide:
                    p.data.mul_(1-lmbda*lr)

                # gradient step
                p.data.addcdiv_(grad_avg, Dk, value=-tau) # x_k - tau*(d_k/D_k)

                # Proximal way of weight decay
                if lmbda > 0 and self.divide:
                    p.data.div_(1+lmbda*lr)

        #############################
        ## Maintenance
        if _use_fstar_this_iter:
            self.state['h'] = h
            self.state['f_star'] = self.lb
        
        self.state['step_size_list'].append(t1)

        return loss