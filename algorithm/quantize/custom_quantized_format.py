"""
In this file, we implement our customized quantized intermediate format (json+weights, saved as pkl)
We provide functions to convert PyTorch models into the pickle format
and also the method to build simulated quantized models from the pickle file
"""
import torch
import torch.nn as nn
import numpy as np
from .quantized_ops import to_np, to_pt, USE_FP_SCALE
from .quantized_ops_diff import QuantizedConv2dDiff as QuantizedConv2d
from .quantized_ops_diff import QuantizedMbBlockDiff as QuantizedMbBlock
from .quantized_ops_diff import QuantizedElementwiseDiff as QuantizedElementwise
from .quantized_ops_diff import QuantizedAvgPoolDiff as QuantizedAvgPool

__all__ = ['build_quantized_network_from_cfg', 'get_effective_scale']


##########################################################
######## PyTorch model to pickle quantized format ########
##########################################################


########################################################################
######## pickle quantized format to simulated quantizated model ########
########################################################################

def get_effective_scale(scale_x, scale_w, scale_y):
    scale_x = to_np(scale_x)
    scale_y = to_np(scale_y)
    # warning: all scales should be double type
    effective_scale = scale_x.astype('double') * scale_w.astype('double') / scale_y.astype('double')
    if USE_FP_SCALE:
        return effective_scale
    else:
        raise NotImplementedError


def build_quantized_conv_from_cfg(conv_cfg, w_bit=8, a_bit=None):
    kwargs = {
        'zero_x': to_pt(conv_cfg['params']['x_zero']),
        'zero_w': to_pt(0),
        'zero_y': to_pt(conv_cfg['params']['y_zero']),
    }

    if USE_FP_SCALE:
        effective_scale = get_effective_scale(conv_cfg['params']['x_scale'], conv_cfg['params']['w_scales'],
                                              conv_cfg['params']['y_scale'])
        kwargs['effective_scale'] = to_pt(effective_scale)
    else:
        raise NotImplementedError
    if isinstance(conv_cfg['kernel_size'], int):  # make tuple
        conv_cfg['kernel_size'] = (conv_cfg['kernel_size'],) * 2
    padding = ((conv_cfg['kernel_size'][0] - 1) // 2, (conv_cfg['kernel_size'][1] - 1) // 2)
    conv = QuantizedConv2d(conv_cfg['in_channel'], conv_cfg['out_channel'], conv_cfg['kernel_size'],
                           padding=padding, stride=conv_cfg['stride'],
                           groups=conv_cfg['groups'], w_bit=w_bit, a_bit=a_bit,
                           **kwargs)
    conv.weight.data = to_pt(conv_cfg['params']['weight'])
    conv.bias.data = to_pt(conv_cfg['params']['bias'])
    # Note that these parameters are added for convenience, not actually needed
    conv.x_scale = conv_cfg['params']['x_scale']
    conv.y_scale = conv_cfg['params']['y_scale']
    return conv


def build_quantized_block_from_cfg(blk_cfg, n_bit=8):
    blk = []
    if blk_cfg['pointwise1'] is not None:
        blk.append(build_quantized_conv_from_cfg(blk_cfg['pointwise1'], w_bit=n_bit))
    if blk_cfg['depthwise'] is not None:
        blk.append(build_quantized_conv_from_cfg(blk_cfg['depthwise'], w_bit=n_bit))
    if 'se' in blk_cfg and blk_cfg['se'] is not None:
        raise NotImplementedError  # TODO: SE backward is not implemented yet
    if blk_cfg['pointwise2'] is not None:
        blk.append(build_quantized_conv_from_cfg(blk_cfg['pointwise2'], w_bit=n_bit))

    if blk_cfg['residual'] is not None:  # with residual connection
        if 'kernel_size' in blk_cfg['residual']:  # the conv case
            scale_x = blk_cfg['residual']['params']['y_scale']
            zero_x = blk_cfg['residual']['params']['y_zero']

            scale_conv = blk_cfg['pointwise2']['params']['y_scale']
            zero_conv = blk_cfg['pointwise2']['params']['y_zero']

            scale_y = blk_cfg['residual']['params']['out_scale']
            zero_y = blk_cfg['residual']['params']['out_zero']

            q_add = QuantizedElementwise('add',
                                         to_pt(to_np(zero_x)),
                                         to_pt(to_np(zero_conv)),
                                         to_pt(to_np(zero_y)),
                                         to_pt(to_np(scale_x)),
                                         to_pt(to_np(scale_conv)),
                                         to_pt(to_np(scale_y)), )
            residual_conv = build_quantized_conv_from_cfg(blk_cfg['residual'], w_bit=n_bit)
        else:
            scale_x = blk_cfg['residual']['params']['x_scale']
            zero_x = blk_cfg['residual']['params']['x_zero']

            scale_conv = blk_cfg['pointwise2']['params']['y_scale']
            zero_conv = blk_cfg['pointwise2']['params']['y_zero']

            scale_y = blk_cfg['residual']['params']['y_scale']
            zero_y = blk_cfg['residual']['params']['y_zero']

            q_add = QuantizedElementwise('add',
                                         to_pt(to_np(zero_x)),
                                         to_pt(to_np(zero_conv)),
                                         to_pt(to_np(zero_y)),
                                         to_pt(to_np(scale_x)),
                                         to_pt(to_np(scale_conv)),
                                         to_pt(to_np(scale_y)), )
            residual_conv = None
    else:
        q_add = None
        residual_conv = None
    return QuantizedMbBlock(nn.Sequential(*blk), q_add, residual_conv, a_bit=n_bit)


def build_quantized_network_from_cfg(cfg, n_bit=8):
    # building the network using our own json format
    if 'first_conv' in cfg:  # ProxylessNAS backbone
        first_conv = build_quantized_conv_from_cfg(cfg['first_conv'], w_bit=n_bit)
    else:
        raise NotImplementedError
    blocks = nn.Sequential(*[build_quantized_block_from_cfg(b, n_bit=n_bit) for b in cfg['blocks']])
    if cfg['feature_mix'] is not None:  # add a feature mix layer
        feature_mix_conv = build_quantized_conv_from_cfg(cfg['feature_mix'], w_bit=n_bit)
    else:
        feature_mix_conv = nn.Identity()
    avgpool = QuantizedAvgPool()
    fc = build_quantized_conv_from_cfg(cfg['classifier'], w_bit=n_bit, a_bit=8)  # always use 8-bit output
    net = nn.Sequential(first_conv, blocks, feature_mix_conv, avgpool, fc)
    return net
