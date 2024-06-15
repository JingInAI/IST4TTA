import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


__all__ = [
    'train_transforms', 'test_transforms',
    'ImageNet', 'ImageNet_C',
]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize])
test_transforms  = transforms.Compose([transforms.Resize(256),
									   transforms.CenterCrop(224),
									   transforms.ToTensor(),
									   normalize])


class ImageNet(ImageFolder):
    num_classes = 1000
    channels = 3
    
    train_transforms = train_transforms
    test_transforms = test_transforms

    DOMAINS = ['train', 'val']

    def __init__(self, root, train=['train'], transform=None, target_transform=None):
        assert len(train) == 1 and train[0] in self.DOMAINS, f'train should be one of {self.DOMAINS}'
        train = train[0] == 'train'
        root = os.path.join(root, 'ImageNet', 'train' if train else 'val')
        super().__init__(root, transform, target_transform)


class ImageNet_C(ImageFolder):
    num_classes = 1000
    channels = 3
    
    train_transforms = train_transforms
    test_transforms = test_transforms
    
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
            target_transform=None
        ):
        assert corruption in self.all_corruptions, f'corruption should be one of {self.all_corruptions}'
        assert level in self.all_levels, f'level should be one of {self.all_levels}'

        root = os.path.join(root, 'ImageNet-C', corruption, str(level))
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.data = []
        self.targets = []
        for path, target in self.samples:
            self.data.append(path)
            self.targets.append(target)
