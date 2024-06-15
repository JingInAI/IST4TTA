import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as _CIFAR10


__all__ = [
    'train_transforms', 'test_transforms',
    'CIFAR10', 'CIFAR10_C',
]

normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize])
test_transforms  = transforms.Compose([transforms.ToTensor(),
                                       normalize])


class CIFAR10(_CIFAR10):
    num_classes = 10
    channels = 3

    train_transforms = train_transforms
    test_transforms = test_transforms

    DOMAINS = ['train', 'test']

    def __init__(self, root, train=['train'], transform=None, target_transform=None, download=False):
        assert len(train) == 1 and train[0] in self.DOMAINS, f'train should be one of {self.DOMAINS}'
        train = train[0] == 'train'
        super().__init__(root, train, transform, target_transform, download)


class CIFAR10_C(CIFAR10):
    all_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
					   'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
					   'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    all_levels = [1, 2, 3, 4, 5]

    def __init__(
            self,
            root: str,
            corruption: str,
            level: int,
            transform=None,
            target_transform=None,
        ):
        assert corruption in self.all_corruptions, f'corruption should be one of {self.all_corruptions}'
        assert level in self.all_levels, f'level should be one of {self.all_levels}'

        super().__init__(root, train=['test'], transform=transform, target_transform=target_transform, download=True)
        raw_data = np.load(os.path.join(root, 'CIFAR-10-C', f'{corruption}.npy'))
        raw_data = raw_data[(level - 1)*10000: level*10000]
        self.data = raw_data
        self.loader = Image.fromarray
