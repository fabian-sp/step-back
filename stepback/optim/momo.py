import torch
import warnings

from ..types import Params, LossClosure, OptFloat

class MoMo(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-1,
                 weight_decay: float=0,
                 beta: float=0.9,
                 lb: float=0,
                 bias_correction: bool=False) -> None:
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if (beta < 0.0) or (beta > 1.0):
            raise ValueError("Invalid beta parameter: {}".format(beta))
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(MoMo, self).__init__(params, defaults)
        
        self.lr = lr
        self.beta = beta # weight for newest element in all averages
        self.lb = lb
        self._weight_decay_flag = (weight_decay > 0)
        
        # how to do exp. averaging
        self.bias_correction = bias_correction
        
        # Initialization
        self._number_steps = 0
        self.state['step_size_list'] = list() # for storing
        
        return
        
    def step(self, closure: LossClosure=None) -> OptFloat:
        
        with torch.enable_grad():
            loss = closure()

        self._number_steps += 1
        beta = self.beta  
        
        ###### Preliminaries
        if self._number_steps == 1:
            if self.bias_correction:
                self.loss_avg = 0.
            else:
                self.loss_avg = loss.detach().clone()
        
        self.loss_avg = beta*self.loss_avg + (1-beta)*loss        

        if self.bias_correction:
            rho = 1-beta**self._number_steps # must be after incrementing k
        else:
            rho = 1
            
        _dot = 0.
        _gamma = 0.
        _norm = 0.
        
        ############################################################
        # Compute all quantities
        # notation in PDF translation:
        # d_k: p.grad_avg, gamma_k: _gamma, \bar f_k: self.loss_avg
        for group in self.param_groups:
            for p in group['params']:
                
                grad = p.grad.data.detach()
                state = self.state[p]

                # Initialize EMA
                if self._number_steps == 1:
                    if self.bias_correction:
                        state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                        state['grad_dot_w'] = torch.zeros(1).to(p.device)
                    else:
                        # Exponential moving average of gradients
                        state['grad_avg'] = grad.clone()
                        # Exponential moving average of inner product <grad, weight>
                        state['grad_dot_w'] = torch.sum(torch.mul(p.data, grad))
                        
                grad_avg, grad_dot_w = state['grad_avg'], state['grad_dot_w']

                grad_avg.mul_(beta).add_(grad, alpha=1-beta)
                grad_dot_w.mul_(beta).add_(torch.sum(torch.mul(p.data, grad)), alpha=1-beta)

                _dot += torch.sum(torch.mul(p.data, grad_avg))
                _gamma += grad_dot_w
                _norm += torch.sum(torch.mul(grad_avg, grad_avg))

        ###### Update weights
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            
            if lmbda > 0:
                nom = (rho+lr*lmbda)/rho * (self.loss_avg - self.lb)  + _dot - (rho+lr*lmbda)/rho * _gamma
                t1 = max(nom, 0.)/_norm
            else:
                t1 = max(self.loss_avg - self.lb + _dot - _gamma, 0.)/_norm
            
            t1 = t1.item() # make scalar
            
            # step size 
            tau = min(lr/rho, t1)
                
            for p in group['params']:   
                state = self.state[p]
                grad_avg = state['grad_avg']          
                p.data.add_(other=grad_avg, alpha=-tau)
                
                if lmbda > 0:
                    p.data.mul_(rho/(rho+lr*lmbda))
                    
        ############################################################
        
        self.state['step_size_list'].append(t1)
        
        return loss
