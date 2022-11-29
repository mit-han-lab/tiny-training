import torch
import torch.nn as nn
import numpy as np

from ..core.utils.config import configs
from ..quantize.quantized_ops_diff import ScaledLinear

def _append_flatten(model_q):
    model_q = list(model_q)
    model_q.append(nn.Flatten())
    model_q = nn.Sequential(*model_q)
    return model_q


def create_scaled_head(model_q, norm_feat=False):
    assert isinstance(model_q, nn.Sequential)
    if not isinstance(model_q[-1], nn.Flatten):
        model_q = _append_flatten(model_q)
    model_q[-2] = ScaledLinear(model_q[-2].in_channels, configs.data_provider.num_classes,
                               model_q[-2].x_scale, model_q[-2].zero_x, norm_feat=norm_feat)
    return model_q


def create_quantized_head(model_q):
    from .quantized_ops_diff import QuantizedConv2dDiff
    assert isinstance(model_q, nn.Sequential)
    if not isinstance(model_q[-1], nn.Flatten):
        model_q = _append_flatten(model_q)
    sample_linear = nn.Conv2d(model_q[-2].in_channels, configs.data_provider.num_classes, 1)

    w_scales = get_weight_scales(sample_linear.weight.data, 8)
    w, b = get_quantized_weight_and_bias(sample_linear.weight.data, sample_linear.bias.data, w_scales,
                                         model_q[-2].x_scale, 8)

    org_op = model_q[-2]
    # here we do not have y_scale, so that the output has the same scale
    effective_scale = (model_q[-2].x_scale * w_scales).float()

    model_q[-2] = QuantizedConv2dDiff(model_q[-2].in_channels, configs.data_provider.num_classes, 1,
                                      zero_x=model_q[-2].zero_x, zero_y=0,  # keep same args
                                      effective_scale=effective_scale,
                                      w_bit=8, a_bit=8,
                                      )
    model_q[-2].weight.data = torch.from_numpy(w).float()
    model_q[-2].bias.data = torch.from_numpy(b).float()
    model_q[-2].x_scale = org_op.x_scale
    model_q[-2].y_scale = 1  # skipped actually
    return model_q


def get_weight_scales(w, n_bit=8, k_near_zero_tolerance=1e-6, allow_all_same=False):
    # NOTICE: the zero point for w is always chosen as 0, so it is actually a symmetric quantization
    def _extract_min_max_from_weight(weights):
        dim_size = weights.shape[0]

        if weights.max() == weights.min():  # all the elements are the same?
            mins = np.zeros(dim_size)
            maxs = np.zeros(dim_size)
            single_value = weights.min().item()
            if single_value < 0.:
                mins[:] = single_value
                maxs[:] = -single_value
            elif single_value > 0.:
                mins[:] = -single_value
                maxs[:] = single_value
            else:
                mins[:] = maxs[:] = single_value
            return torch.from_numpy(mins).to(weights.device), torch.from_numpy(maxs).to(weights.device)
        else:
            weights = weights.reshape(weights.shape[0], -1)
            mins = weights.min(dim=1)[0]
            maxs = weights.max(dim=1)[0]
            maxs = torch.max(mins.abs(), maxs.abs())
            mins = -maxs
            return mins, maxs

    def _expand_very_small_range(mins, maxs):
        k_smallest_half_range = k_near_zero_tolerance / 2
        if (maxs - mins).min() > k_near_zero_tolerance:
            return mins, maxs
        else:
            for i in range(len(mins)):
                mins[i] = min(mins[i], -k_smallest_half_range)
                maxs[i] = max(maxs[i], k_smallest_half_range)
            return mins, maxs

    mins, maxs = _extract_min_max_from_weight(w)
    mins, maxs = _expand_very_small_range(mins, maxs)
    assert (mins + maxs).max() < 1e-9  # symmetric
    return maxs / (2 ** (n_bit - 1) - 1)


def get_quantized_weight_and_bias(w, b, w_scales, x_scale, n_bit=8):
    w = w / w_scales.view(-1, 1, 1, 1)
    w = w.round().int()
    assert w.min().item() >= - 2 ** (n_bit - 1) + 1 and w.max().item() <= 2 ** (n_bit - 1) - 1
    w = w.cpu().numpy().astype(np.int8)

    b = b / w_scales / x_scale
    b = b.round().int()
    assert b.min().item() >= -2147483648 and b.max().item() <= 2147483647
    b = b.cpu().numpy().astype(np.int32)

    return w, b


