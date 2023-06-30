"""
This is the Imagenet32 loader. In order to use, download the Imagenet32 dataset from the official website.

Then make sure to extract 
    * the train set to the directory data/imagenet32/train. It should contain ten files with names <train_data_batch_i>.
    * the validation set to the directory data/imagenet32/val. It should contain one file with name <val_data>.

"""
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Normalize, Compose, RandomCrop, RandomHorizontalFlip, ToTensor

if sys.version_info[0] == 2:
    import _pickle as cPickle
else:
    import pickle

class Imagenet32(VisionDataset):
    """
    Loads the data.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, cuda=False, sz=32):

        super(Imagenet32, self).__init__(root, 
                                         transform=transform,
                                         target_transform=target_transform
                                         )
        self.base_folder = root
        self.train = train  # training set or validation set
        self.cuda = cuda

        self.data = []
        self.targets = []

        # now load the pickled numpy arrays
        if self.train:
            for i in range(1, 11):
                file_name = 'train_data_batch_' + str(i)
                self._add_entry(file_name)
        else:
            file_name = 'val_data'
            self._add_entry(file_name)       
                
        self.targets = [t - 1 for t in self.targets]
        self.data = np.vstack(self.data).reshape(-1, 3, sz, sz)
        
        if self.cuda:
            self.data = torch.FloatTensor(self.data).half().cuda()  # type(torch.cuda.HalfTensor)
        else:
            self.data = self.data.transpose((0, 2, 3, 1))

    def _add_entry(self, file_name):
        file_path = os.path.join(self.base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

        return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.cuda:
            img = self.transform(img)
            return img, target

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_imagenet32(split, path):

    if split.lower() not in ('train', 'val'):
        raise ValueError(f"split must be in ('train', 'val'), but got {split}")
    
    # data directory
    traindir = os.path.expanduser(os.path.join(path, "imagenet32/train/"))
    valdir = os.path.expanduser(os.path.join(path, "imagenet32/val/"))
    
    # normalization
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    train = (split.lower() != 'val')
    if train:
        transform = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]) 
    else:
        transform =Compose([
           ToTensor(),
            normalize,
        ]) 

    dataset = Imagenet32(
        traindir if train else valdir,
        train=train,
        transform=transform)

    return dataset

