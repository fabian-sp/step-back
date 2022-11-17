import torch
import warnings

from ..types import Params, LossClosure, OptFloat

class MoMo(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-3,
                 beta: float=0.9,
                 lb: float=0) -> None:
        
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if (beta < 0.0) or (beta > 1.0):
            raise ValueError("Invalid beta parameter: {}".format(beta))
        
        defaults = dict(lr=lr)
        
        super(MoMo, self).__init__(params, defaults)
        
        self.lr = lr
        self.beta = beta # weight for newest element in all averages
        self.lb = lb
        
        self.loss_avg = None # average over the loss evaluations
        # flag that we did step() at least once
        self._flag_first_step = False
        
        # initialize averages
        for group in self.param_groups:
            for p in group['params']:
                p.grad_avg = torch.zeros(p.shape)
                p.grad_dot_w = 0.
        
        self.state['step_size_list'] = list() # for storing
        
        return
        
    def step(self, closure: LossClosure=None) -> OptFloat:
        
        with torch.enable_grad():
            loss = closure()
        
        # update loss average
        if self.loss_avg is None:
            self.loss_avg = loss
        else:
            self.loss_avg = self.beta*loss + (1-self.beta)*self.loss_avg
            
        if self._flag_first_step:
            beta = self.beta
        else:
            beta = 1
                
        _dot = 0.
        _gamma = 0.
        _norm = 0.
        
        ############################################################
        # compute all quantities
        for group in self.param_groups:
            for p in group['params']:
                
                p.grad_avg = beta*p.grad + (1-beta)*p.grad_avg
                p.grad_dot_w = beta*torch.sum(torch.mul(p.data, p.grad)) + (1-beta)*p.grad_dot_w
                
                _dot += torch.sum(torch.mul(p.data, p.grad_avg))
                _gamma += p.grad_dot_w
                _norm += torch.sum(torch.mul(p.grad_avg, p.grad_avg))
                
        t1 = max(self.loss_avg + _dot - _gamma , self.lb)/_norm
        t1 = t1.item() # make scalar
        
        # update weights
        for group in self.param_groups:
            lr = group['lr']
            
            for p in group['params']:              
                tau = min(lr, t1) # step size                
                p.data.add_(other=p.grad_avg, alpha=-tau)
                
        ############################################################
        self._flag_first_step = True
        self.state['step_size_list'].append(t1)
        
        return loss