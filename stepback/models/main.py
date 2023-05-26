import torch
import warnings

from .basic_models import MLP, MatrixFac, MatrixComplete
from .vgg import get_cifar_vgg
from .resnet import get_cifar_resnet
from .kuangliu_resnet import get_kuangliu_resnet
from .vit import ViT, SimpleViT

def get_model(config: dict={}) -> torch.nn.Module:
    """
    Main function mapping a model name to an instance of torch.nn.Module.  
    """
    kwargs = config['model_kwargs']
    name = config['model']
    
    #======== Linear model =============
    if name == 'linear':
        assert len(config['_input_dim']) == 1, "Expecting input dimensionality of length 1."
        
        input_size = config['_input_dim'][0]
        output_size = config['model_kwargs'].get('output_size', 1)

        model = MLP(input_size=input_size, output_size=output_size, hidden_sizes=[], bias=False)
    
    #======== MLP with ReLU =============
    elif name == 'mlp':
        assert len(config['_input_dim']) == 1, "Expecting input dimensionality of length 1."
        input_size = config['_input_dim'][0]
        
        assert 'output_size' in config['model_kwargs'].keys(), "Need to specify the dimension of the output. Add in your config \n 'model_kwargs' = {'output_size': }"
        output_size = config['model_kwargs'] # output of model can be multi-dim, but targets are 1-dim 
        
        model = MLP(input_size=input_size, **kwargs)
    
    #======== Matrix factorization =============
    elif name == 'matrix_fac':
        assert len(config['_input_dim']) == 1, "Expecting input dimensionality of length 1."
        assert len(config['_output_dim']) == 1, "Expecting output dimensionality of length 1."
        
        input_size = config['_input_dim'][0]
        output_size = config['_output_dim'][0]
        
        if 'rank' not in kwargs.keys():
            warnings.warn(f'No rank dimension specified. Using max of input and output size, equal to {max(input_size, output_size)}')
        
        model = MatrixFac(input_size=input_size, output_size=output_size, rank=kwargs.get('rank', max(input_size, output_size)))
    
    #======== Matrix completion =============
    elif name == 'matrix_completion':
        assert len(config['_input_dim']) == 1, "Expecting input dimensionality of length 1."
        assert len(config['_output_dim']) == 1, "Expecting output dimensionality of length 1."
        
        if 'rank' not in kwargs.keys():
            warnings.warn(f'No rank dimension specified. Using default of 10.')
        
        model = MatrixComplete(dim1=kwargs['dim'][0], dim2=kwargs['dim'][1], rank=kwargs.get('rank', 10))
    
    #======== VGG models =============
    elif name in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        
        if config['dataset'] == 'cifar10':
            model = get_cifar_vgg(name, num_classes=10, **kwargs) # batch_norm False by default
        elif config['dataset'] == 'cifar100':
            model = get_cifar_vgg(name, num_classes=100, **kwargs) # batch_norm False by default
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")   
    
    #======== Resnet =============
    elif name in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
        
        if config['dataset'] == 'cifar10':
                model = get_cifar_resnet(name, num_classes=10, **kwargs) # batch_norm True by default
        elif config['dataset'] == 'cifar100':
                model = get_cifar_resnet(name, num_classes=100, **kwargs) # batch_norm True by default
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")   
    
    elif name in ['resnet18-kuangliu', 'resnet34-kuangliu', 'resnet50-kuangliu', 'resnet101-kuangliu', 'resnet152-kuangliu']:
        if config['dataset'] == 'imagenet32':
            model = get_kuangliu_resnet(name, num_classes=1000)
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")
    
     #======== Vision transformer =============
    elif name == 'vit':
        num_classes = 10 if config['dataset'] == 'cifar10' else 100
        model = ViT(image_size = 32, num_classes = num_classes,**kwargs)   
    else:
        raise KeyError(f"Unknown model option {name}.")   
    
    return model
        
        