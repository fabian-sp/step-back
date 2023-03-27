import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings

from .momo import MoMo
from .sps import SPS

def get_optimizer(opt_config: dict) -> (torch.optim.Optimizer, dict):
    """
    Main function mapping opt configs to an instance of torch.optim.Optimizer and a dict of hyperparameter arguments (lr, weight_decay,..).  
    
    For all hyperparameters which are not specified, we use PyTorch default.
    """
    
    name = opt_config['name']
    
    if opt_config.get('lr') is None:
        warnings.warn("You have not specified a learning rate. A default value of 1e-3 will be used.")
    
    if name == 'sgd':
        opt_obj = torch.optim.SGD
        
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0)
                  }
        
    elif name == 'sgd-m':
        opt_obj = torch.optim.SGD
        # sgd-m with exp. weighted average should have dampening = momentum
        if opt_config.get('dampening') == 'momentum':
            dampening = opt_config.get('momentum', 0.9)
        else:
            dampening = opt_config.get('dampening', 0)
            
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': False,
                  'dampening': dampening
                  }

    elif name == 'sgd-nesterov':
        opt_obj = torch.optim.SGD
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': True,
                  'dampening': opt_config.get('dampening', 0)
                  }
               
    elif name == 'adam':
        opt_obj = torch.optim.Adam
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8)
                  }
    
    elif name == 'adamw':
        opt_obj = torch.optim.AdamW
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8)
                  }
    
    elif name == 'momo':
        opt_obj = MoMo
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False)
                  }
        
    elif name == 'prox-sps':
        opt_obj = SPS
        hyperp = {'lr': opt_config.get('lr', 1e-3),
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'lb': opt_config.get('lb', 0.),
                  'prox': True
                  }
        
    else:
        raise KeyError(f"Unknown optimizer name {name}.")
        
    return opt_obj, hyperp

def get_scheduler(config: dict, opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.
    """
    # if not specified, use constant step sizes
    name = config.get('lr_schedule', 'constant')
    
    if name == 'constant':
        lr_fun = lambda epoch: 1 # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
    
    elif name == 'linear':
        lr_fun = lambda epoch: 1/(epoch+1) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'sqrt':
        lr_fun = lambda epoch: (epoch+1)**(-1/2) # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)
        
    elif name == 'exponential':
        # TODO: allow arguments
        scheduler = StepLR(opt, step_size=50, gamma=0.5)
        
    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")
    
    return scheduler