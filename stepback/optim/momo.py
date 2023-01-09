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
                 bias_correction: bool=True) -> None:
        
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
        
        self._number_steps = 0
        
        # initialize averages
        self.loss_avg = 0
        for group in self.param_groups:
            for p in group['params']:
                p.grad_avg = torch.zeros(p.shape).to(p.device)
                p.grad_dot_w = 0.
        
        self.state['step_size_list'] = list() # for storing
        
        return
        
    def step(self, closure: LossClosure=None) -> OptFloat:
        
        with torch.enable_grad():
            loss = closure()
            
        if self._number_steps >= 1:
            beta = self.beta
        else:
            if self.bias_correction:
                beta = self.beta
            else:
                beta = 0. # first iter, use all quantities with coeff. 1
        
        ###### Preliminaries
        self.loss_avg = (1-beta)*loss +  beta*self.loss_avg                
        self._number_steps += 1
        if self.bias_correction:
            rho = 1-beta**self._number_steps # must be after incrementing k
        else:
            rho = 1
            
        _dot = 0.
        _gamma = 0.
        _norm = 0.
        
        ############################################################
        # compute all quantities
        # notation in PDF translation:
        # d_k = p.grad_avg, gamma_k = _gamma, \bar f_k = self.loss_avg
        for group in self.param_groups:
            for p in group['params']:
                
                p.grad_avg = (1-beta)*p.grad + beta*p.grad_avg
                p.grad_dot_w = (1-beta)*torch.sum(torch.mul(p.data, p.grad)) + beta*p.grad_dot_w
                
                _dot += torch.sum(torch.mul(p.data, p.grad_avg))
                _gamma += p.grad_dot_w
                _norm += torch.sum(torch.mul(p.grad_avg, p.grad_avg))
                      
        ###### Update weights
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            
            # \bar f_k + <d_k, x^k> - gamma_k
            if lmbda > 0:
                nom = (rho+lr*lmbda)/rho * (self.loss_avg - self.lb)  + _dot - (rho+lr*lmbda)/rho * _gamma
                t1 = max(nom, 0.)/_norm
            else:
                t1 = max(self.loss_avg - self.lb + _dot - _gamma, 0.)/_norm
            
            t1 = t1.item() # make scalar
            
            # step size 
            tau = min(lr/rho, t1)
                
            for p in group['params']:                             
                p.data.add_(other=p.grad_avg, alpha=-tau)
                
                if lmbda > 0:
                    p.data.mul_(rho/(rho+lr*lmbda))
                    
        ############################################################
        
        self.state['step_size_list'].append(t1)
        
        return loss
