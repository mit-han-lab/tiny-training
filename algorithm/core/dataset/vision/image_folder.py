import os
from typing import Callable, Dict, Optional

from torchvision import datasets
import warnings
import random

__all__ = ['ImageFolder']


class ImageFolerFilterWarning(datasets.ImageFolder):
    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader=datasets.folder.default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            return super().__getitem__(index)


class ImageFolder(dict):
    def __init__(self,
                 root: str,
                 transforms: Optional[Dict[str, Callable]] = None,
                 target_transforms: Optional[Dict[str, Callable]] = None) -> None:
        if transforms is None:
            transforms = {'train': None, 'val': None}
        if target_transforms is None:
            target_transforms = {'train': None, 'val': None}

        super().__init__({
            split: ImageFolerFilterWarning(root=os.path.join(root, split),
                                           transform=transforms[split],
                                           target_transform=target_transforms[split])
            for split in ['train', 'val']
        })
