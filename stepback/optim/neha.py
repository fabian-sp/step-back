"""
Author: Robert Gower & Fabian Schiapp

Adapted from https://github.com/fabian-sp/ProxSPS/blob/main/sps/sps.py.

Main changes:

"""

import torch
import warnings

from ..types import Params, LossClosure, OptFloat
# NumErical High-order Adaptive learning rate
class NeHa(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=0.001,
                 r: float=0.5,
                 weight_decay: float=0, 
                 theta: float=1.0, 
                 lr_max: float=1.0,
                 lr_min: float=1e-6)-> None:
        """
        
        Parameters
        ----------
        params : 
            PyTorch model parameters.
        lr: learnign rate.
        r : tolerance in residual. The default is 0.5.
        weight_decay : float, optional
            Weigt decay parameter. The default is 0.
            If specified, the term weight_decay/2 * ||w||^2 is added to objective, where w are all model weights.
        theta : float, optional
            gain parameter
        lr_max: float, optional
            max stepsize
        lr_min: float, optional
            min stepsize            
            
        """
        
        params = list(params)
        defaults = dict(lr = lr,
                        r=r,
                        weight_decay=weight_decay, 
                        theta=theta, 
                        lr_max=lr_max,
                        lr_min=lr_min)
        
        super(NeHa, self).__init__(params, defaults)
        self.params = params
        
        self.lr = lr
        self.theta = theta
        self.r = r
        self.lr_max = lr_max
        self.lr_min = lr_min
        # initialize
        self._number_steps = 0
        self.state['step_size_list'] = list()
        
        if len(self.param_groups) > 1:
            warnings.warn("More than one parameter group.")
        
        return
        
    def step(self, closure: LossClosure=None) -> OptFloat:
        """
        NEHA update
        See https://opt-ml.org/papers/2021/paper28.pdf.
        """
        self._number_steps +=1
        with torch.enable_grad():
            loss = closure()
        
        # NOTE: we could implement the "proximal version" similar to AdamW?
        reg = self.add_weightdecay()
        loss.add_(reg)  # Adding l2-norm directly to loss

        # Initializing the high order iterate
        if self._number_steps == 1:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    x = p.data.detach()
                    state['xH'] = x.clone() #torch.clone(p.data, memory_format=torch.preserve_format).detach()      
        ############################################################
        # First pass through parameters to compute SGD update
        for group in self.param_groups:
            lr = group['lr']
            # lmbda = group['weight_decay']
            for p in group['params']:
                state = self.state[p]
                grad = p.grad.data.detach()
                p.data.add_(other=grad, alpha=-lr)  
                # xH = state['xH']      
                # xH.add_(grad, alpha=-0.5*lr)

        # gradient = torch.autograd.grad(loss, model.parameters())
        with torch.enable_grad():
            loss = closure()
        reg = self.add_weightdecay()
        loss.add_(reg)  # Adding l2-norm directly to loss
        # loss.backward(retain_graph=True)  # Compute new gradient? NOTE: Doesn't work!!!
        # Second pass to finish computing high-order update and next stepsize
        delta =0
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                state = self.state[p]
                grad = p.grad.data.detach() #NOTE: because we didn't zero gradient, this has the sum of both gradients!
                xH = state['xH']      
                xH.add_(grad, alpha=-0.5*lr)
                delta += ((xH-p.data)**2).sum()

        # Update learning rate
        for group in self.param_groups:
            group['lr'] = group['lr'] * (self.r/torch.sqrt(delta))**(self.theta)
        ############################################################       
        # update state with metrics
        self.state['step_size_list'].append(lr) # works only if one param_group!

        return loss
    
    

    @torch.no_grad()
    def add_weightdecay(self):
        reg = 0          
        for group in self.param_groups:
            lmbda = group['weight_decay']
            for p in group['params']:
                p.grad.add_(lmbda * p.data)  # gradients
                reg += (lmbda/2) * (p.data**2).sum() # loss
        return reg


