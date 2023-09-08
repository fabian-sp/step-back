import os

from stepback.record import Record
from stepback.utils import get_output_filenames

os.chdir('..')
print("Saving records from directory: ", os.getcwd())


ALL_EXP = ['cifar100_resnet110', 'cifar10_resnet20', 'cifar10_vgg16', 'cifar10_vit', 'mnist_mlp', 'imagenet32_resnet18']

for exp_id in ALL_EXP:

    print(f"============= Saving record for {exp_id} =============")
    output_names = get_output_filenames(exp_id)
    R = Record(output_names)


    R.to_csv(name=exp_id)