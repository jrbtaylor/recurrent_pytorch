"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import torch
from torchvision import datasets, transforms

dtype = torch.cuda.FloatTensor

pixel_permutation = torch.randperm(28*28)

def mnist(batch_size,sequential=True,permuted=True,n_workers=4):
    if permuted:  # can't be permuted if not sequential
        assert sequential

    kwargs = {'num_workers': n_workers, 'pin_memory': True}
    if not sequential:
        transform = transforms.ToTensor()
    elif not permuted:
        transform = transforms.Compose([
            transforms.ToTensor(),transforms.Lambda(lambda x: x.view(-1, 1))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),transforms.Lambda(
                lambda x: x.view(-1, 1)[pixel_permutation])])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/media/jason/DataSSD1/data/mnist',
                       train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/media/jason/DataSSD1/data/mnist',
                       train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader,val_loader