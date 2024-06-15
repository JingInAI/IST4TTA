import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from .tools import pil_loader, broadcast_tensor



class UnlabeledDataset():
    def __init__(self, datasets, class_names=None, preprocess=None, keep_in_memory=True):
        """
        Args:
            dataset (list): List of datasets, each dataset is a object of HuggingFace Dataset class
            class_names (list): List of class names
            preprocess (function): Preprocess function to be applied to each image
            keep_in_memory (bool): Keep the dataset in memory or not
        """
        self.samples = []
        self.labels = []    # Ground truth labels, which stored the class names
        self.class_names = [] if class_names is None else class_names
        self.custom_classes = False if class_names is None else True
        self.keep_in_memory = keep_in_memory

        datasets = datasets if isinstance(datasets, list) else [datasets]
        
        for d in datasets:
            if self.keep_in_memory:
                if 'image' in d.features:
                    self.samples.extend(d['image'])
                elif 'img' in d.features:
                    self.samples.extend(d['img'])
                else:
                    raise NotImplementedError('No image feature found in dataset')
                
                if 'label' in d.features:
                    labels = d['label']
                    class_names_ = d.features['label'].names
                elif 'fine_label' in d.features:
                    labels = d['fine_label']
                    class_names_ = d.features['fine_label'].names
                else:
                    raise NotImplementedError('No label feature found in dataset')

                labels = [class_names_[l] for l in labels]
                self.labels.extend(labels)
            else:
                self.samples = d

                if 'image' in d.features:
                    self.labels.append('image')
                elif 'img' in d.features:
                    self.labels.append('img')
                else:
                    raise NotImplementedError('No image feature found in dataset')

                if 'label' in d.features:
                    self.labels.append('label')
                    class_names_ = d.features['label'].names
                elif 'fine_label' in d.features:
                    self.labels.append('fine_label')
                    class_names_ = d.features['fine_label'].names
                else:
                    raise NotImplementedError('No label feature found in dataset')

            if class_names is None:
                self.class_names.extend(class_names_)

        self.preprocess = preprocess
        self.pseudo_labels = None   # Pseudo labels, which stored the index of {self.class_names}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        assert len(self.samples) == len(self.pseudo_labels), 'Length of samples and pseudo labels must be the same'
        
        sample = self.samples[idx]
        pseudo_label = self.pseudo_labels[idx]

        if not isinstance(sample, Image.Image):
            sample = pil_loader(sample)

        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample, pseudo_label

    def set_pseudo_labels(self, define_func, **kwargs):
        """ Set pseudo labels for the dataset
        Args:
            define_func (function): Function to define pseudo labels
        """
        self.pseudo_labels = define_func(self.samples, self.class_names, **kwargs)
        self.pseudo_labels_accuracy()
        
    def pseudo_labels_accuracy(self):
        """ Compute the accuracy of pseudo labels
        """
        if not self.custom_classes and dist.get_rank() == 0:
            self.acc = sum([1 if l == self.class_names[p.argmax(-1)] else 0 for l, p in zip(self.labels, self.pseudo_labels)]) / len(self)
            print(f'Default classes. Accuracy of pseudo labels: {self.acc:.4f}')





class UnlabeledDatasetV2(UnlabeledDataset):
    def __init__(self, dataset, preprocess=None):
        """
        Args:
            dataset (torchvision.datasets.ImageFolder): Dataset
            preprocess (function): Preprocess function to be applied to each image
        """
        self.samples = []
        self.labels = []
        self.class_names = []

        for path, label in dataset.samples:
            self.samples.append(path)
            self.labels.append(label)
        
        self.preprocess = preprocess
        self.pseudo_labels = None

    def pseudo_labels_accuracy(self):
        """ Compute the accuracy of pseudo labels
        """
        if dist.get_rank() == 0:
            self.acc = sum([1 if l == p.argmax(-1) else 0 for l, p in zip(self.labels, self.pseudo_labels)]) / len(self)
            print(f'Accuracy of pseudo labels of curract batch: {self.acc:.4f}')





class UnlabeledDatasetV3(UnlabeledDataset):
    def __init__(self, dataset, extend=1):
        """
        Args:
            dataset (torchvision.datasets.ImageFolder): Dataset
            preprocess (function): Preprocess function to be applied to each image
            extend (int): Extend the dataset by {extend} times
        """
        assert extend >= 1, 'extend must be greater than 1'
        
        self.samples = []
        self._samples = []
        self.labels = []
        self.verbose = True

        for path, label in dataset.samples:
            self.samples.append(path)
            self.labels.append(label)
            self._samples.extend([path] * extend)
        
        self.extend = extend
        self.class_names = dataset.class_names if hasattr(dataset, 'class_names') else []
    
    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        assert len(self._samples) == len(self._pseudo_labels), 'Length of samples and pseudo labels must be the same'
        
        sample = self._samples[idx]
        pseudo_label = self._pseudo_labels[idx]
        soft_label = self._soft_labels[idx]

        return sample, pseudo_label, soft_label
    
    def set_pseudo_labels(self, define_func, **kwargs):
        """ Set pseudo labels for the dataset
        Args:
            define_func (function): Function to define pseudo labels
        """
        pseudo_label_dataset = define_func(self.samples, self.class_names, **kwargs)

        self._samples = pseudo_label_dataset.samples
        self._features = pseudo_label_dataset.features
        self._pseudo_labels = pseudo_label_dataset.pseudo_labels
        self._soft_labels = pseudo_label_dataset.soft_labels
        self._hard_labels = pseudo_label_dataset.hard_labels

        labels = []
        for l in self.labels:
            labels.extend([l] * pseudo_label_dataset.extend)
        self._labels = labels

        self.pseudo_labels_accuracy()

    def pseudo_labels_accuracy(self):
        """ Compute the accuracy of pseudo labels
        """
        if dist.get_rank() == 0:
            soft_labels = np.array(self._soft_labels)
            pseudo_labels = np.array(self._pseudo_labels)

            predict_labels = soft_labels * pseudo_labels
            C = predict_labels.shape[-1]
            predict_labels = predict_labels.reshape(-1, self.extend * C)
            predict_labels = np.argmax(predict_labels, axis=-1) % C

            self.acc = sum([1 if l == p else 0 for l, p in zip(self.labels, predict_labels)]) / len(self.samples)
            if self.verbose:
                print(f'Accuracy of pseudo labels of curract batch: {self.acc:.4f}')




class Trans(object):
    def __init__(self, samples, labels, extend):
        self.samples = samples
        self.labels = labels
        self.extend = extend
        
        self.targets = []
        for e in self.samples:
            self.targets.append(e[self.labels[1]])
    
    def __getitem__(self, idx):
        idx = idx // self.extend
        example = self.samples[idx]
        image = example[self.labels[0]]
        return image
    
    def __len__(self):
        return len(self.samples) * self.extend



class UnlabeledDatasetV4(UnlabeledDataset):
    def __init__(self, dataset, extend=1, keep_in_memory=True):
        assert extend >= 1, 'extend must be greater than 1'
        super().__init__(dataset, keep_in_memory=keep_in_memory)

        self._samples = []
        self.extend = extend
        self.keep_in_memory = keep_in_memory
        self.verbose = True

        if self.keep_in_memory:
            for s in self.samples:
                self._samples.extend([s] * extend)

            labels = []
            for l in self.labels:
                labels.append(self.class_names.index(l))
            self.labels = labels
        else: 
            self._samples = Trans(self.samples, self.labels, self.extend)
            self.labels = self._samples.targets
    
    def __len__(self):
        return len(self._samples)
    
    def __getitem__(self, idx):
        assert len(self._samples) == len(self._pseudo_labels), 'Length of samples and pseudo labels must be the same'
    
        pseudo_label = self._pseudo_labels[idx]
        soft_label = self._soft_labels[idx]

        if self._locations is None:
            sample = self._samples[idx]
        else:
            ori_sample = self._samples[idx]
            location = self._locations[idx]
    
            if isinstance(ori_sample, str):
                ori_sample = pil_loader(ori_sample)
            assert isinstance(ori_sample, Image.Image), 'Sample must be PIL image or path to image'

            imgh, imgw, i, j, h, w, f = location
            sample = F.resized_crop(ori_sample, i, j, h, w, (imgh, imgw), self._preprocess[0].interpolation)
            if f == 1:
                sample = F.hflip(sample)
            
            sample = self._preprocess[1](sample)

        return sample, pseudo_label, soft_label

    def set_pseudo_labels(self, define_func, **kwargs):
        pseudo_label_dataset = define_func(self.samples, self.class_names, **kwargs)

        self._locations = pseudo_label_dataset.locations
        if self._locations is None:
            self._samples = pseudo_label_dataset.samples
            if isinstance(self._samples, PseudoTrans):
                self._samples = self._samples.sample_list
        else:
            self._preprocess = pseudo_label_dataset.preprocess

        self._features = pseudo_label_dataset.features
        self._pseudo_labels = pseudo_label_dataset.pseudo_labels
        self._soft_labels = pseudo_label_dataset.soft_labels
        self._hard_labels = pseudo_label_dataset.hard_labels

        labels = []
        for l in self.labels:
            labels.extend([l] * pseudo_label_dataset.extend)
        self._labels = labels

        self.pseudo_labels_accuracy()

    def pseudo_labels_accuracy(self):
        if dist.get_rank() == 0:
            soft_labels = np.array(self._soft_labels)
            pseudo_labels = np.array(self._pseudo_labels)

            predict_labels = soft_labels * pseudo_labels
            C = predict_labels.shape[-1]
            predict_labels = predict_labels.reshape(-1, self.extend * C)
            predict_labels = np.argmax(predict_labels, axis=-1) % C

            self.acc = sum([1 if l == p else 0 for l, p in zip(self.labels, predict_labels)]) / len(self.samples)
            if self.verbose:
                print(f'Accuracy of pseudo labels of curract batch: {self.acc:.4f}')




class UnlabeledDatasetV5(UnlabeledDatasetV3):
    def __init__(self, batch, extend=1, verbose=True):
        """
        Args:
            batch (tuple): Tuple of images and labels
            extend (int): Extend the dataset by copying the samples
        """
        assert extend >= 1, 'extend must be greater than 1'

        self.samples = []
        self._samples = []
        self.labels = []
        self.class_names = []

        images, labels = batch
        for image, label in zip(images, labels):
            image = transforms.ToPILImage()(image)
            self.samples.append(image)
            self.labels.append(label)
            self._samples.extend([image] * extend)

        self.extend = extend
        self.verbose = verbose

    def get_features(self):
        if '_features' not in self.__dict__:
            return None
        else:
            return self._features

    def get_soft_labels(self):
        if '_soft_labels' not in self.__dict__:
            return None
        else:
            return self._soft_labels.tolist()

    def get_pseudo_labels(self):
        if '_pseudo_labels' not in self.__dict__:
            return None
        else:
            return self._pseudo_labels





class UnlabeledDatasetV6(UnlabeledDatasetV4):
    def __init__(self, batch, class_names=None, extend=1, verbose=True):
        assert extend >= 1, 'extend must be greater than 1'

        self.samples = []
        self._samples = []
        self.labels = []
        self.class_names = class_names

        images, labels = batch['img'], batch['label']
        for image, label in zip(images, labels):
            assert isinstance(image, Image.Image)
            self.samples.append(image)
            self.labels.append(label)
            self._samples.extend([image] * extend)
        
        self.extend = extend
        self.verbose = verbose

    def get_features(self):
        if '_features' not in self.__dict__:
            return None
        else:
            return self._features
    
    def get_soft_labels(self):
        if '_soft_labels' not in self.__dict__:
            return None
        else:
            return self._soft_labels.tolist()
    
    def get_pseudo_labels(self):
        if '_pseudo_labels' not in self.__dict__:
            return None
        else:
            return self._pseudo_labels





class PseudoTrans(object):
    def __init__(self, samples, extend):
        self.samples = samples
        self.extend = extend
        self.sample_list = [None for _ in range(len(self))]

    def __getitem__(self, idx):
        idx = idx // self.extend
        example = self.samples[idx]
        if 'image' in example.keys():
            image = example['image']
        elif 'img' in example.keys():
            image = example['img']
        else:
            raise Exception('Unknown key for image')
        return image

    def __len__(self):
        return len(self.samples) * self.extend
    
    def set_sample(self, idx, sample):
        self.sample_list[idx] = sample



class PseudoLabelDataset():
    def __init__(self, samples, class_names, preprocess=None, extend=1, device='cuda', verbose=True):
        """
        Args:
            samples (list): List of samples, each sample is PIL image
            class_names (list): List of class names
            preprocess (function): Preprocess function to be applied to each image
            extend (int): Extend the dataset by copying the samples
            device (str): Device to store the features
            verbose (bool): Print verbose information
        """
        self.class_names = class_names
        self.preprocess = preprocess
        self.extend = extend
        self.device = device
        self.keep_in_memory = True if isinstance(samples, list) else False

        assert extend >= 1, 'Extend must be greater than 1'

        if self.keep_in_memory:
            self.samples = []
            for s in samples:
                self.samples.extend([s] * extend)
        else:
            self.samples = PseudoTrans(samples, extend)

        self.pseudo_labels = [None for _ in range(len(self.samples))]
        self.features = [None for _ in range(len(self.samples))]
        self.locations = None

        if dist.get_rank() == 0 and verbose:
            print(f'Pseudo label dataset original size: {len(samples)}, extended size: {len(self)} ({extend}x)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        loc = ()

        if not isinstance(sample, torch.Tensor):

            if isinstance(sample, str):
                sample = pil_loader(sample)
            assert isinstance(sample, Image.Image), 'Sample must be PIL image or path to image'

            if self.preprocess is not None:
                if isinstance(self.preprocess, list) and len(self.preprocess) == 2:
                    sample, loc = self.preprocess[0](sample)
                    sample = self.preprocess[1](sample)
                else:
                    sample = self.preprocess(sample)

        return sample, idx, loc

    def set_samples(self, indexes, samples):
        """ Set samples for indexes
        Args:
            indexes (list): List of indexes
            samples (list): List of samples
        """
        assert len(indexes) == len(samples), 'Length of indexes and samples must be the same'
        for i, s in zip(indexes, samples):
            if self.keep_in_memory:
                self.samples[i] = s
            else:
                self.samples.set_sample(i, s)
        
    def set_locations(self, indexes, locations):
        """ Set locations for samples
        Args:
            indexes (list): List of indexes
            locations (list): List of locations, (hei, wid, x, y, w, h, flip)
        """
        assert len(indexes) == len(locations), "Length of indexes and locations must be the same"
        if self.locations is None:
            self.locations = [None for _ in range(len(self.samples))]
        for i, loc in zip(indexes, locations):
            self.locations[i] = loc
    
    def set_features(self, indexes, features):
        """ Set features for samples
        Args:
            indexes (list): List of indexes
            features (list): List of features
        """
        assert len(indexes) == len(features), 'Length of indexes and features must be the same'
        for i, f in zip(indexes, features):
            self.features[i] = f

    def set_pseudo_labels(self, indexes, labels):
        """ Set labels for samples
        Args:
            indexes (list): List of indexes
            labels (list): List of labels
        """
        assert len(indexes) == len(labels), 'Length of indexes and labels must be the same'
        for i, l in zip(indexes, labels):
            self.pseudo_labels[i] = l

    def get_clean_labels(self):
        """ Get clean labels with size of len(self.samples) / self.extend
        Returns:
            clean_labels (np.array)
        """
        clean_labels = []
        for i in range(len(self) // self.extend):
            labels = self.pseudo_labels[i*self.extend : (i+1)*self.extend]
            labels = np.array(labels).mean(axis=0)
            clean_labels.append(labels)
        
        return np.array(clean_labels)

    def broadcast_pseudo_labels(self, src):
        """ Broadcast labels to all GPUs
        """
        pseudo_labels = torch.tensor(np.array(self.pseudo_labels), dtype=torch.float32).to(self.device)
        pseudo_labels = broadcast_tensor(pseudo_labels, src=src)
        self.set_pseudo_labels(list(range(len(pseudo_labels))), pseudo_labels.cpu().numpy())





class MemoryBank():
    def __init__(self, max_len=None):
        self.max_len = max_len
        self.features = []
        self.labels = []

    def update(self, features: list, labels: list):
        assert len(features) == len(labels), 'Length of features and labels must be the same'
        self.features.extend(features)
        self.labels.extend(labels)

        if self.max_len is not None and len(self.features) > self.max_len:
            if self.max_len <= 0:
                self.features = []
                self.labels = []
            else:
                self.features = self.features[-self.max_len:]
                self.labels = self.labels[-self.max_len:]

    def get_all(self):
        return self.features, self.labels

    def size(self):
        assert len(self.features) == len(self.labels), 'Length of features and labels must be the same'
        return len(self.features)
