"""
Author: Fabian Schaipp

Adapted from https://github.com/fabian-sp/ProxSPS/blob/main/sps/sps.py.

Main changes:
    * use .data in all computations
    * rename 'fstar' to 'lb'
"""

import torch
import warnings

from ..types import Params, LossClosure, OptFloat

class SPS(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-3,
                 weight_decay: float=0, 
                 lb: float=0, 
                 prox: bool=True)-> None:
        """
        
        Parameters
        ----------
        params : 
            PyTorch model parameters.
        lr : float, optional
            Learning rate. The default is 1e-3.
        weight_decay : float, optional
            Weigt decay parameter. The default is 0.
            If specified, the term weight_decay/2 * ||w||^2 is added to objective, where w are all model weights.
        fstar : float, optional
            Lower bound of loss function. The default is 0 (which is a lower bound for most loss functions).
        prox: bool, optional
            Whether to use ProxSPS or SPS.
            
        """
        
        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(SPS, self).__init__(params, defaults)
        self.params = params
        
        self.lr = lr
        self.lb = lb
        self.prox = prox

        self.state['step_size_list'] = list()
        
        if len(self.param_groups) > 1:
            warnings.warn("More than one parameter group for SPS.")
        
        return
        
    def step(self, closure: LossClosure=None) -> OptFloat:
        """
        ProxSPS update

        See https://arxiv.org/abs/2301.04935.
        """
        
        with torch.enable_grad():
            loss = closure()
        
        # get lower bound of objective
        lb = self.lb
        
        # add l2-norm if not ProxSPS
        if not self.prox:
            r = 0          
            for group in self.param_groups:
                lmbda = group['weight_decay']
                for p in group['params']:
                    p.grad.add_(lmbda * p.data)  # gradients
                    r += (lmbda/2) * (p.data**2).sum() # loss
                    
            loss.add_(r) 
        
                
        if self.prox:
            grad_norm, grad_dot_w = self.compute_grad_terms(need_gdotw=True)
        else:
            grad_norm, _ = self.compute_grad_terms(need_gdotw=False)
            
        ############################################################
        # update 
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            
            for p in group['params']:

                if self.prox:
                    nom = (1+lr*lmbda)*(loss - lb) - lr*lmbda*grad_dot_w
                else:
                    nom = loss - lb
                    
                denom = (grad_norm)**2 
                t1 = (nom/denom).item()
                t2 = max(0., t1)                 
                
                # compute tau^+
                tau = min(lr, t2) 
                
                p.data.add_(other=p.grad.data, alpha=-tau)
                if self.prox:
                    p.data.div_(1+lr*lmbda)
            
        ############################################################       
        # update state with metrics
        self.state['step_size_list'].append(t2) # works only if one param_group!

        return loss
    
    @torch.no_grad()
    def compute_grad_terms(self, need_gdotw=True):
        """
        computes:
            * norm of stochastic gradient ||grad||
            * inner product <grad,param> (needed only for prox=True). 
        """
        grad_norm = 0.
        grad_dot_w = 0.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    raise KeyError("None gradient")
                
                g = p.grad.data
                grad_norm += torch.sum(torch.mul(g, g))
                if need_gdotw:
                    grad_dot_w += torch.sum(torch.mul(p.data, g))
          
        grad_norm = torch.sqrt(grad_norm)
        return grad_norm, grad_dot_w
    




