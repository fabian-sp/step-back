import torch
import copy

"""
Multi-layer perceptron with ReLU activations.
"""
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

"""
Model for matrix factorization.
"""
def MatrixFac(input_size: int, output_size: int, rank: int):
    """
    Model has form:
        W2 @ W1
    
    """
    W1 = torch.nn.Linear(input_size, rank, bias=False)
    W2 = torch.nn.Linear(rank, output_size, bias=False)
    
    return torch.nn.Sequential(W1, W2)

"""
Model for matrix completion.
"""    
class MatrixComplete(torch.nn.Module):
    """
    Model has form:
        U.T @ V + b_U[:,None] + b_V[None,:]
        
    U: (dim1,rank)
    V: (dim2,rank)
    b_U: (dim1,)
    b_V: (dim2,)
    
    Hence, b_U and b_V are added as rank-one matrices (all columns/rows equal).
    """
    def __init__(self, dim1: int, dim2: int, rank: int):
        super(MatrixComplete, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
        self.U = torch.nn.Linear(dim1, rank, bias=False)
        self.V = torch.nn.Linear(dim2, rank, bias=False)
        
        # bias parameters (can be seen as rank-one matrices)
        self.bias_U = torch.nn.Parameter(torch.randn(dim1))
        self.bias_V = torch.nn.Parameter(torch.randn(dim2))
        
    def forward(self, x):
        # x contains [row of U, column of V]        
        x1 = torch.nn.functional.one_hot(x[:,0].long(), self.dim1).float()
        x2 = torch.nn.functional.one_hot(x[:,1].long(), self.dim2).float()
        
        prod = torch.diag(self.U(x1) @ self.V(x2).T).reshape(-1)
        b1 = x1 @ self.bias_U
        b2 = x2 @ self.bias_V
        
        return (prod + b1 + b2)[:,None] # [batch_size,1] output
    
    def get_matrix(self):
        W = self.U.weight.T @ self.V.weight + self.bias_U[:,None] + self.bias_V[None,:]
        return W
        