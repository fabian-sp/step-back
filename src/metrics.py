import torch 

# Loss functions (used in training or validation)
def get_metric_function(metric: str):
    
    if metric == "cross_entropy":
        l = cross_entropy_loss

    elif metric == "logistic":
        l = logistic_loss

    elif metric == "squared":
        l = squared_loss
    
    elif metric == "logistic_accuracy":
        l = logistic_accuracy

    elif metric == "cross_entropy_accuracy":
        l = cross_entropy_accuracy
    
    else:
        raise KeyError(f"Unknown loss function {metric}.")
    
    return l


    
def cross_entropy_loss(out, targets, backwards=False):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, targets.long().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

# one-dimensional output expected, use .view(-1)
def logistic_loss(out, targets, backwards=False):
    criterion = torch.nn.SoftMarginLoss()
    loss = criterion(out.view(-1), targets.float().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

# squared loss is invariant under .view(-1)
def squared_loss(out, targets, backwards=False):
    criterion = torch.nn.MSELoss()
    loss = criterion(out.view(-1), targets.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


##
# Accuracy
# ==========================

def logistic_accuracy(out, targets):
    logits = torch.sigmoid(out).view(-1)
    pred_labels = ((logits>=0.5)*2-1).view(-1) # map to{-1,1}
    acc = (pred_labels == targets).float().mean()
    return acc

def cross_entropy_accuracy(out, targets):
    pred_labels = out.argmax(dim=1)
    acc = (pred_labels == targets).float().mean()
    return acc
