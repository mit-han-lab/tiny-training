import torch
from typing import List
import math
from ..utils import dist
from ..utils.config import configs

__all__ = ['build_lr_scheduler', 'CosineLRwithWarmup']


def build_lr_scheduler(optimizer, batch_per_epoch):
    if configs.run_config.lr_schedule_name == 'cosine':
        lr_scheduler = CosineLRwithWarmup(
            optimizer, configs.run_config.warmup_epochs * batch_per_epoch, configs.run_config.warmup_lr * dist.size(),
            configs.run_config.n_epochs * batch_per_epoch,
            final_lr=configs.run_config.get('final_lr', 0) * dist.size()
        )
    elif configs.run_config.lr_schedule_name == 'step':
        lr_scheduler = StepLRwithWarmup(
            optimizer, configs.run_config.warmup_epochs * batch_per_epoch, configs.run_config.warmup_lr * dist.size(),
            configs.run_config.get('lr_step_size', 30) * batch_per_epoch,
            configs.run_config.get('lr_step_gamma', 0.1)
        )
    else:
        raise NotImplementedError(configs.run_config.lr_schedule_name)
    return lr_scheduler


class CosineLRwithWarmup(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self,
                 optimizer,
                 warmup_steps: int,
                 warmup_lr: float,
                 decay_steps: int,
                 final_lr: float = 0.,
                 last_epoch: int = -1) -> None:
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.decay_steps = decay_steps
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * self.last_epoch / self.warmup_steps + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            current_steps = self.last_epoch - self.warmup_steps
            return [
                0.5 * (base_lr - self.final_lr) * (1 + math.cos(math.pi * current_steps / self.decay_steps)) + self.final_lr
                for base_lr in self.base_lrs
            ]


class StepLRwithWarmup(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self,
                 optimizer,
                 warmup_steps: int,
                 warmup_lr: float,
                 step_size: int,
                 gamma: float = 0.1,
                 last_epoch: int = -1) -> None:
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * self.last_epoch / self.warmup_steps + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            current_steps = self.last_epoch - self.warmup_steps
            n_decay = current_steps // self.step_size
            return [
                base_lr * (self.gamma ** n_decay)
                for base_lr in self.base_lrs
            ]
