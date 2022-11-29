from .vision import *
from ..utils.config import configs
from .vision.transform import *
import torchvision

__all__ = ['build_dataset']


def build_dataset():
    if configs.data_provider.dataset == 'image_folder':
        dataset = ImageFolder(
            root=configs.data_provider.root,
            transforms=ImageTransform(),
        )
    elif configs.data_provider.dataset == 'imagenet':
        dataset = ImageNet(root=configs.data_provider.root,
                       transforms=ImageTransform(), )
    elif configs.data_provider.dataset == 'cifar10':
        dataset = {
            'train': torchvision.datasets.CIFAR10(configs.data_provider.root, train=True,
                                                  transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.CIFAR10(configs.data_provider.root, train=False,
                                                transform=ImageTransform()['val'], download=True),
        }
    elif configs.data_provider.dataset == 'cifar100':
        dataset = {
            'train': torchvision.datasets.CIFAR100(configs.data_provider.root, train=True,
                                                   transform=ImageTransform()['train'], download=True),
            'val': torchvision.datasets.CIFAR100(configs.data_provider.root, train=False,
                                                 transform=ImageTransform()['val'], download=True),
        }
    elif configs.data_provider.dataset == 'imagehog':
        dataset = ImageHog(
            root=configs.data_provider.root,
            transforms=ImageTransform(),
        )
    else:
        raise NotImplementedError(configs.data_provider.dataset)

    return dataset
