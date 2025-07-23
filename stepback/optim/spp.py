"""
Author: Fabian Schaipp
"""

import torch
import warnings

from ..types import Params, LossClosure, OptFloat

class SPP(torch.optim.Optimizer):
    def __init__(self, 
                 params: Params, 
                 lr: float=1e-3,
                 weight_decay: float=0, 
                 )-> None:
        """

        Parameters
        ----------
        params : 
            Model parameters.
        lr : float, optional
            Learning rate. The default is 1e-3.
        weight_decay : float, optional
            Weigt decay parameter. The default is 0.
        """
        
        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(SPP, self).__init__(params, defaults)
        self.params = params
        
        self.lr = lr

        warnings.warn("SPP is applicable only to linear regression problems!")
        
        return
    
    def prestep(self, out, data, targets, ind, loss_name):
        
        self._A = data.clone().detach()
        self._r = targets.clone().detach()

        assert loss_name == 'squared'

        return
        
    def step(self, closure: LossClosure=None) -> OptFloat:
        """
        SPP update for a linear regression problem.
        """
        
        with torch.enable_grad():
            loss = closure()
        
        assert len(self.param_groups) == 1, "For linear regression, we only expect one param group."

        ############################################################
        # update 
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            bs = self._A.shape[0]
            
            assert len(group['params']) == 1, "For linear regression, we only expect one param tensor."
            for p in group['params']:
                mat = (1/bs) * (self._A.T @ self._A) +  (1/lr + lmbda) * torch.eye(self._A.shape[1]).to(self._A.device)
                rhs = (1/lr) * p.data + (1/bs) * (self._A.T @ self._r)
                sol = torch.linalg.solve(mat, rhs.T)
                # p needs shape [1, dim]
                p.data.copy_(sol.T)
                
        ############################################################       
        # 
        return loss




