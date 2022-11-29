import torch
from typing import Union, List, Any
import torch.distributed
import numpy as np
from core.utils import dist

__all__ = ['ddp_reduce_tensor', 'DistributedMetric', 'accuracy', 'AverageMeter']


def list_sum(x: List) -> Any:
    r"""
    return the sum of a list of objects (can be int, float, torch.Tensor, np.ndarray, etc)
    can be used for adding losses
    """
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: List) -> Any:
    r"""
    return the mean of a list of objects (can be int, float, torch.Tensor, np.ndarray, etc)
    """
    return list_sum(x) / len(x)


def ddp_reduce_tensor(tensor: torch.Tensor, reduce='mean') -> Union[torch.Tensor, List[torch.Tensor]]:
    if dist.size() == 1:
        return tensor
    tensor_list = [
        torch.empty_like(tensor) for _ in range(dist.size())
    ]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == 'mean':
        return list_mean(tensor_list)
    elif reduce == 'sum':
        return list_sum(tensor_list)
    elif reduce == 'cat':
        return torch.cat(tensor_list, dim=0)
    else:
        return tensor_list


class DistributedMetric(object):
    r"""
    average metrics for distributed training.
    """

    def __init__(self, name: str, backend='ddp'):
        self.name = name
        self.sum = 0
        self.count = 0
        self.backend = backend

    def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
        val *= delta_n
        if type(val) in [int, float]:
            val = torch.Tensor(1).fill_(val).cuda()
        if self.backend == 'ddp':
            self.count += ddp_reduce_tensor(torch.Tensor(1).fill_(delta_n).cuda(), reduce='sum')
            self.sum += ddp_reduce_tensor(val.detach(), reduce='sum')
        else:
            raise NotImplementedError

    @property
    def avg(self):
        if self.count == 0:
            return torch.Tensor(1).fill_(-1)
        else:
            return self.sum / self.count


class AverageMeter(object):
    r"""
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[torch.Tensor, np.ndarray, float, int], n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    r"""
    Computes the precision@k for the specified values of k
    """
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k <= output.shape[1]:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(torch.zeros(1).cuda() - 1.)
    return res
