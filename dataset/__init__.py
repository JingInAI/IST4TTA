from .cifar import CIFAR10, CIFAR10_C
from .imagenet import ImageNet, ImageNet_C


def load_dataset(name, path, domains, preprocess=None):
    if name == 'cifar10':
        return CIFAR10(root=path, train=domains, download=True, transform=preprocess)
    elif name == 'cifar10-c':
        return CIFAR10_C(root=path, corruption=domains[0], level=domains[1], transform=preprocess)
    elif name == 'imagenet':
        return ImageNet(root=path, train=domains, transform=preprocess)
    elif name == 'imagenet-c':
        return ImageNet_C(root=path, corruption=domains[0], level=domains[1], transform=preprocess)
    else:
        raise NotImplementedError


def get_transforms(name):
    if name in ['cifar10', 'cifar10-c']:
        return CIFAR10.train_transforms, CIFAR10.test_transforms
    elif name in ['imagenet', 'imagenet-c']:
        return ImageNet.train_transforms, ImageNet.test_transforms
    else:
        raise NotImplementedError
