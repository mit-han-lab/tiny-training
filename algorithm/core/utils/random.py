r"""
Use torch.Generator() to control randomness
"""
import torch
from typing import List, Any, Optional, Union

__all__ = ['torch_random_choices', 'torch_randint', 'torch_random', 'torch_shuffle', 'torch_uniform']


def torch_random_choices(src_list: List[Any], generator: Optional[torch.Generator], k=1) -> Union[Any, List[Any]]:
    rand_idx = torch.randint(low=0, high=len(src_list), generator=generator, size=(k,))
    out_list = [src_list[i] for i in rand_idx]

    return out_list[0] if k == 1 else out_list


def torch_randint(low: int, high: int, generator: Optional[torch.Generator]) -> int:
    if low == high:
        return low
    else:
        assert low < high
        return int(torch.randint(low=low, high=high, generator=generator, size=(1,)))


def torch_random(generator: Optional[torch.Generator]) -> float:
    return float(torch.rand(1, generator=generator))


def torch_shuffle(src_list: List[Any], generator: Optional[torch.Generator]) -> List[Any]:
    rand_indexes = torch.randperm(len(src_list), generator=generator).tolist()
    return [
        src_list[i] for i in rand_indexes
    ]


def torch_uniform(a: Union[int, float], b: Union[int, float], generator: Optional[torch.Generator]) -> float:
    rand_val = torch_random(generator)
    return (b - a) * rand_val + a
