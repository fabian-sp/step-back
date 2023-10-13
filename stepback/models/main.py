import torch
import warnings
from torchvision.models import resnet18, resnet50

from .basic_models import MLP, MatrixFac, MatrixComplete
from .vgg import get_cifar_vgg
from .resnet import get_cifar_resnet
from .kuangliu_resnet import get_kuangliu_resnet


def get_num_classes(dataset_name):
    if dataset_name == 'cifar10':
        C = 10
    elif dataset_name == 'cifar100':
        C = 100
    elif dataset_name in ['imagenet', 'imagenet32']:
        C = 1000
    else:
        raise KeyError(f"Unknown number of classes for dataset {dataset_name}.")

    return C


def get_model(config: dict, input_dim: list, output_dim: list) -> torch.nn.Module:
    """
    Main function mapping a model name to an instance of torch.nn.Module.  
    """
    kwargs = config['model_kwargs']
    name = config['model']
    
    #======== Linear model =============
    if name == 'linear':
        assert len(input_dim) == 1, "Expecting input dimensionality of length 1."
        
        input_size = input_dim[0]
        output_size = config['model_kwargs'].get('output_size', 1)
        bias = config['model_kwargs'].get('bias', False)

        model = MLP(input_size=input_size, output_size=output_size, hidden_sizes=[], bias=bias)
    
    #======== MLP with ReLU =============
    elif name == 'mlp':
        assert len(input_dim) == 1, "Expecting input dimensionality of length 1."
        input_size = input_dim[0]
        
        assert 'output_size' in config['model_kwargs'].keys(), "Need to specify the dimension of the output. Add in your config \n 'model_kwargs' = {'output_size': }"
        output_size = config['model_kwargs'] # output of model can be multi-dim, but targets are 1-dim 
        
        model = MLP(input_size=input_size, **kwargs)
    
    #======== Matrix factorization =============
    elif name == 'matrix_fac':
        assert len(input_dim) == 1, "Expecting input dimensionality of length 1."
        assert len(output_dim) == 1, "Expecting output dimensionality of length 1."
        
        input_size = input_dim[0]
        output_size = output_dim[0]
        
        if 'rank' not in kwargs.keys():
            warnings.warn(f'No rank dimension specified. Using max of input and output size, equal to {max(input_size, output_size)}')
        
        model = MatrixFac(input_size=input_size, output_size=output_size, rank=kwargs.get('rank', max(input_size, output_size)))
    
    #======== Matrix completion =============
    elif name == 'matrix_completion':
        assert len(input_dim) == 1, "Expecting input dimensionality of length 1."
        assert len(output_dim) == 1, "Expecting output dimensionality of length 1."
        
        if 'rank' not in kwargs.keys():
            warnings.warn(f'No rank dimension specified. Using default of 10.')
        
        model = MatrixComplete(dim1=kwargs['dim'][0], dim2=kwargs['dim'][1], rank=kwargs.get('rank', 10))
    
    #======== VGG models =============
    elif name in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        
        if config['dataset'] in ['cifar10', 'cifar100']:
            num_classes = get_num_classes(config['dataset'])
            model = get_cifar_vgg(name, num_classes=num_classes, **kwargs) # batch_norm False by default
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")   
    
    #======== Resnet =============
    elif name in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
        
        if config['dataset'] in ['cifar10', 'cifar100']:
            num_classes = get_num_classes(config['dataset'])
            model = get_cifar_resnet(name, num_classes=num_classes, **kwargs) # batch_norm True by default
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")   
    
    elif name in ['resnet18-kuangliu', 'resnet34-kuangliu', 'resnet50-kuangliu', 'resnet101-kuangliu', 'resnet152-kuangliu']:
        if config['dataset'] == 'imagenet32':
            model = get_kuangliu_resnet(name, num_classes=1000)
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")
        
    elif name in ['resnet18-pytorch', 'resnet50-pytorch']:
        if config['dataset'] == 'imagenet':     
            if name == 'resnet18-pytorch':
                model = resnet18(pretrained=False)
            elif name == 'resnet50-pytorch':
                model = resnet50(pretrained=False)
        else:
            raise KeyError(f"Model {name} is not implemented yet for dataset {config['dataset']}.")
    
    # ======== Vision transformer =============
    elif name == 'vit':
        from .vit import ViT    # lazy import because of einops

        num_classes = get_num_classes(config['dataset'])
        model = ViT(image_size=32, num_classes=num_classes,**kwargs)   
    
    elif name == 'swint':
        from .vit import swin_t # lazy import because of einops

        num_classes = get_num_classes(config['dataset'])
        model = swin_t(num_classes=num_classes,
                       downscaling_factors=(2,2,2,1),
                       **kwargs)   
    
    else:
        raise KeyError(f"Unknown model option {name}.")   
    
    return model
        
        