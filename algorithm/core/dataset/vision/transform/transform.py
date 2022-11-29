import torch
import math
from core.utils.config import configs
from torchvision import transforms
import random

__all__ = ['ImageTransform']


class ImageTransform(dict):
    def __init__(self):
        super().__init__({
            'train': self.build_train_transform(),
            'val': self.build_val_transform()
        })

    def build_train_transform(self):
        if 'vww' in configs.data_provider.root:
            t = transforms.Compose([
                transforms.Resize((configs.data_provider.image_size, configs.data_provider.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std),
            ])
        else:
            from timm.data import create_transform
            t = create_transform(
                input_size=configs.data_provider.image_size,
                is_training=True,
                color_jitter=configs.data_provider.color_aug,
                mean=self.mean_std['mean'],
                std=self.mean_std['std'],
            )

        return t
    
    def build_val_transform(self):
        if 'vww' in configs.data_provider.root:
            return transforms.Compose([
                transforms.Resize((configs.data_provider.image_size, configs.data_provider.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(math.ceil(configs.data_provider.image_size / 0.875))),
                transforms.CenterCrop(configs.data_provider.image_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std)
            ])

    @property
    def mean_std(self):
        if True:  # MCU side model
            print('Using MCU transform (leading to range -128, 127)')
            return {'mean': [0.5, 0.5, 0.5], 'std': [1 / 255, 1 / 255, 1 / 255]}
        else:
            return configs.data_provider.get('mean_std',
                                             {'mean': [0.5, 0.5, 0.5],
                                              'std': [0.5, 0.5, 0.5]})
