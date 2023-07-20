"""
Loss and accuracy function objects.

NOTE:
    
    * We generally use reduction='mean' for all loss functions.
    * When using the squared loss (MSELoss), we flatten output and target. 
      Due to reduction='mean', this computes the avergae error, per tensor entry and over the batch.
      E.g. with batch size b and output dimension m, the loss is compute as 1/(b*m) sum (output-target)^2.
      Actually, this is also what Pytorch does internally.
       
"""
import torch 


class Loss:
    def __init__(self, name : str, backwards: bool=False):
        """A loss function object. Can be used either as training loss or as evaluation loss/metric.
        
        The loss can be computed by ```self.compute(out, targets)```.

        Parameters
        ----------
        name : str
            The name of the loss. See below for options.
        backwards : bool, optional
            Whether backpropagation is done in self.compute. 
            Should be only set True for the training losss object. By default False.
        """
        self.name = name
        self.backwards = backwards
        
        # defaults
        self._flatten_target = True
        self._flatten_out = False
        
        if self.name == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        
        elif self.name == 'logistic':
            self.criterion = torch.nn.SoftMarginLoss()
            self._flatten_out = True
        
        elif self.name == 'squared':
            self.criterion = torch.nn.MSELoss()
            self._flatten_out = True
            
        elif self.name == 'cross_entropy_accuracy':
            assert not self.backwards, "For accuracy metrics, we never want to do backprop."
            self.criterion = cross_entropy_accuracy
                  
        elif self.name == 'logistic_accuracy':
            assert not self.backwards, "For accuracy metrics, we never want to do backprop."
            self.criterion = logistic_accuracy
        
        return           
        
    def compute(self, out, targets):
        
        if self._flatten_out:
            out = out.view(-1)
        
        if self._flatten_target:
            targets = targets.view(-1)
            
        loss = self.criterion(out, targets)

        if self.backwards and loss.requires_grad:
            loss.backward()
        
        return loss



##
# Accuracy functions
# ==========================

def logistic_accuracy(out, targets):
    logits = torch.sigmoid(out).view(-1)
    pred_labels = (logits >= 0.5)*2 - 1  # map to{-1,1}
    acc = (pred_labels == targets).float().mean()
    return acc

def cross_entropy_accuracy(out, targets):
    pred_labels = out.argmax(dim=1)
    acc = (pred_labels == targets).float().mean()
    return acc
