import os
import copy
import yaml
import torch
import torch.nn as nn

from ..utils import dist

from ..utils.config import configs
from ..utils.logging import logger
from ..utils.network import remove_bn, trainable_param_num

__all__ = ['BaseTrainer']


class BaseTrainer(object):

    def __init__(self, model: nn.Module, data_loader, criterion, optimizer, lr_scheduler):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion

        self.best_val = 0.0
        self.start_epoch = 0

        # optimization-related
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @property
    def checkpoint_path(self):
        return os.path.join(configs.run_dir, 'checkpoint')

    def save(self, epoch=0, is_best=False):
        if dist.rank() == 0:
            checkpoint = {
                'state_dict': self.model.module.state_dict()
                if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model.state_dict(),
                'epoch': epoch,
                'best_val': self.best_val,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }

            os.makedirs(self.checkpoint_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(self.checkpoint_path, 'ckpt.pth'))

            if is_best:
                torch.save(checkpoint, os.path.join(self.checkpoint_path, 'ckpt.best.pth'))

    def resume(self):
        model_fname = os.path.join(self.checkpoint_path, 'ckpt.pth')
        if os.path.exists(model_fname):
            checkpoint = torch.load(model_fname, map_location='cpu')

            # load checkpoint
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                logger.info('loaded epoch: %d' % checkpoint['epoch'])
            else:
                logger.info('!!! epoch not found in checkpoint')
            if 'best_val' in checkpoint:
                self.best_val = checkpoint['best_val']
                logger.info('loaded best_val: %f' % checkpoint['best_val'])
            else:
                logger.info('!!! best_val not found in checkpoint')
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info('loaded optimizer')
            else:
                logger.info('!!! optimizer not found in checkpoint')
            if 'lr_scheduler' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                logger.info('loaded lr_scheduler')
            else:
                logger.info('!!! lr_scheduler not found in checkpoint')
        else:
            logger.info('Skipping resume... checkpoint not found')

    def validate(self):
        raise NotImplementedError

    def train_one_epoch(self, epoch):
        raise NotImplementedError

    def run_training(self):
        val_info_dict = None
        for epoch in range(self.start_epoch, configs.run_config.n_epochs + configs.run_config.warmup_epochs):
            train_info_dict = self.train_one_epoch(epoch)
            logger.info(f'epoch {epoch}: f{train_info_dict}')

            if (epoch + 1) % configs.run_config.eval_per_epochs == 0 \
                    or epoch == configs.run_config.n_epochs + configs.run_config.warmup_epochs - 1:
                val_info_dict = self.validate()

                is_best = val_info_dict['val/top1'] > self.best_val
                self.best_val = max(val_info_dict['val/top1'], self.best_val)
                if is_best:
                    logger.info(' * New best acc (epoch {}): {:.2f}'.format(epoch, self.best_val))
                val_info_dict['val/best'] = self.best_val
                logger.info(f'epoch {epoch}: {val_info_dict}')
                
                # save model
                self.save(
                    epoch=epoch,
                    is_best=is_best,
                )
        if configs.run_config.grid_output is not None and dist.rank()==0:
            with open(configs.run_config.grid_output, 'a') as f:
                f.write(f'{configs.run_config.grid_ckpt_path}\t{configs.run_config.n_epochs}\t{configs.run_config.bs256_lr}\t{round(self.best_val, 2)}\n')
        # patch for None return bug (not sure why it raises actually...)
        if val_info_dict is None:
            val_info_dict = self.validate()
            val_info_dict['val/best'] = max(val_info_dict['val/top1'], self.best_val)

        return val_info_dict
