
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from filelock import FileLock



def get_cifar10_dataloader(train_bs = 256, test_bs = 256):

    root_dir = '/Users/linjiayi/saddle-free-opt/datasets/cifar10'


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    with FileLock(os.path.expanduser("~/datasets.cifar10.lock")):

        trainset = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=transform_test)
        
    train_size = int(0.8 * len(trainset))
    validation_size = len(trainset) - train_size
    train_dataset, validation_dataset = random_split(trainset, [train_size, validation_size])

    train_loader = DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True)
    valid_loader = DataLoader(
        validation_dataset, batch_size=test_bs, shuffle=False
    )
    test_loader = DataLoader(
        testset, batch_size=test_bs, shuffle=False)

    return train_loader, valid_loader, test_loader




def get_cifar10_dataset():
    root_dir = 'datasets/cifar10'


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    with FileLock(os.path.expanduser("~/datasets.cifar10.lock")):

        trainset = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=transform_test)
        

    return trainset, testset






def get_cifar100_dataset():
    root_dir = '/workspace/datasets/cifar100'


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    with FileLock(os.path.expanduser("~/datasets.cifar100.lock")):

        trainset = datasets.CIFAR100(
            root=root_dir, train=True, download=True, transform=transform_train)

        testset = datasets.CIFAR100(
            root=root_dir, train=False, download=True, transform=transform_test)
        

    return trainset, testset


