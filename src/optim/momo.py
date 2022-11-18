import torch
import warnings

from ..types import Params, LossClosure, OptFloat

class MoMo(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-1,
                 beta: float=0.9,
                 lb: float=0,
                 bias_correction: bool=True) -> None:
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if (beta < 0.0) or (beta > 1.0):
            raise ValueError("Invalid beta parameter: {}".format(beta))
        
        defaults = dict(lr=lr)
        
        super(MoMo, self).__init__(params, defaults)
        
        self.lr = lr
        self.beta = beta # weight for newest element in all averages
        self.lb = lb
        
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
                
        self.loss_avg = (1-beta)*loss +  beta*self.loss_avg        
        
        self._number_steps += 1
        
        _dot = 0.
        _gamma = 0.
        _norm = 0.
        
        ############################################################
        # compute all quantities
        for group in self.param_groups:
            for p in group['params']:
                
                p.grad_avg = (1-beta)*p.grad + beta*p.grad_avg
                p.grad_dot_w = (1-beta)*torch.sum(torch.mul(p.data, p.grad)) + beta*p.grad_dot_w
                
                _dot += torch.sum(torch.mul(p.data, p.grad_avg))
                _gamma += p.grad_dot_w
                _norm += torch.sum(torch.mul(p.grad_avg, p.grad_avg))
                
        # import pdb; pdb.set_trace()
        t1 = max(self.loss_avg + _dot - _gamma , self.lb)/_norm
        t1 = t1.item() # make scalar
        
        # update weights
        for group in self.param_groups:
            lr = group['lr']
            
            # step size 
            if self.bias_correction:
                tau = min(lr/(1-beta**self._number_steps), t1)
            else:
                tau = min(lr, t1) 
                
            
            for p in group['params']:                             
                p.data.add_(other=p.grad_avg, alpha=-tau)
                
        ############################################################
        
        self.state['step_size_list'].append(t1)
        
        return loss