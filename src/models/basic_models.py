import torch
import copy

def MLP(input_size: int, output_size: int, hidden_sizes: list=[], bias: bool=False, dropout: bool=False):
    modules = []
    _hidden = copy.deepcopy(hidden_sizes)
    
    _hidden.insert(0, input_size)
    for i, layer in enumerate(_hidden[:-1]):
        modules.append(torch.nn.Linear(layer, _hidden[i+1],  bias=bias))

        modules.append(torch.nn.ReLU())
        if dropout:
            modules.append(torch.nn.Dropout(p=0.5))

    modules.append(torch.nn.Linear(_hidden[-1], output_size, bias=bias))

    return torch.nn.Sequential(*modules)