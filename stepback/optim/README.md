## Optimizers

If you want to add an optimizer to `step-back`, you need to register it in [main.py](main.py).

* Your optimizer should inherit from `torch.optim.Optimizer`.
* The `step`-method should have one argument called `closure` (if other arguments exists, they will be called with their default). See the method `train_epoch` in [`base.py`](../base.py).


The `closure` argument is a function that takes model output and targets, does `.backward()` and computes (and returns) the loss function value. Your optimizer might need additional information in the step method (e.g. the info of the batch indices).
For this, you can add to your optimizer a method called `prestep`. In `prestep` you can assign attributes which you need later in `step`. It will always be executed before `step`.
For possible arguments, see the below example.


``` python

class MyOptimizer(torch.optim.Optimizer)
    
    def __init__(self):
        return


    def step(self, closure):
        """ Your step method goes here."""
        
        #default way of computing loss and gradient
        with torch.enable_grad():
            loss = closure()

        return loss

    # This method is optional. Only needed if your optimizer requires more than loss value and gradients for doing a step.
    def prestep(out, targets, ind, loss_name):
        """
        out : model output (before loss)
        targets: for computing the loss (e.g. image classes)
        ind: indices of the batch members (from 1,...,n_train_samples)
        loss_name: a string that indicates which loss is used. See `step-back/metrics.py`.
        """

        return
```
