import torchvision
from torchvision import transforms


def get_mnist(split, path, normalize=True, as_image=True):
   
    transform_list = [torchvision.transforms.ToTensor()]
    
    if normalize:
        # from Pytorch examples: https://github.com/pytorch/examples/blob/main/mnist/main.py
        norm = transforms.Normalize((0.1307,), (0.3081,))
        transform_list.append(norm)
    
    # input shape (28,28) or (784,)
    if not as_image:
        view = torchvision.transforms.Lambda(lambda x: x.view(-1).view(784))
        transform_list.append(view)
    
    ds = torchvision.datasets.MNIST(root=path, 
                                    train= (split=='train'),
                                    download=True,
                                    transform=torchvision.transforms.Compose(transform_list))  
    
    return ds