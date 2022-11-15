import torch
import warnings

from .basic_models import MLP, MatrixFac
from .vgg import get_cifar10_vgg
from .resnet import get_cifar10_resnet

def get_model(config: dict={}) -> torch.nn.Module:
    """
    Main function mapping a model name to an instance of torch.nn.Module.  
    """
    kwargs = config['model_kwargs']
    name = config['model']
    
    if name == 'linear':
        assert len(config['_input_dim']) == 1, "Expecting input dimensionality of length 1."
        
        input_size = config['_input_dim'][0]
        model = MLP(input_size=input_size, output_size=1, hidden_sizes=[], bias=False, **kwargs)
    
    elif name == 'matrix_fac':
        assert len(config['_input_dim']) == 1, "Expecting input dimensionality of length 1."
        assert len(config['_output_dim']) == 1, "Expecting output dimensionality of length 1."
        
        input_size = config['_input_dim'][0]
        output_size = config['_output_dim'][0]
        
        if 'rank' not in kwargs.keys():
            warnings.warn(f'No rank dimension specified. Using max of input and output size, equal to {max(input_size, output_size)}')
        
        model = MatrixFac(input_size=input_size, output_size=output_size, rank=kwargs.get('rank', max(input_size, output_size)))
    
    elif name in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        
        if config['dataset'] == 'cifar10':
            model = get_cifar10_vgg(name, **kwargs) # batch_norm False by default
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")   
    
    elif name in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
        
        if config['dataset'] == 'cifar10':
                model = get_cifar10_resnet(name, **kwargs) # batch_norm True by default
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")   
    
    
    else:
        raise KeyError(f"Unknown model option {name}.")   
    
    return model
        
        