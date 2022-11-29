from pprint import pprint
from pydoc import visiblename
import numpy as np

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relay
from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor
from tvm.relay.op.reduce import sum as _sum
from tvm.relay.op import nn as _nn
from tvm.relay.op.tensor import (
    shape_of,
)
from tvm.relay.op.transform import (
    broadcast_to_like,
    collapse_sum_like,
    cast_like,
    reshape,
    reshape_like,
    strided_slice,
    take,
    transpose,
    where,
    repeat,
    expand_dims,
    full_like,
    split,
    squeeze,
    strided_set,
    arange,
    scatter_nd,
)

# import graphviz
from .op2grad import register_gradient, GRAD_OP_MAP
from .diff_ops_bakup import *
from .diff_ops_bakup import _get_reduce_axis, _unreduce_expand

# from graph_tools.visualize_call import visualize_call
def check_call_info(call):
    expr = relay.Function(relay.analysis.all_vars(call), call)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    return mod["main"].body.checked_type


@register_gradient("split")
def split_grad(orig, grad):
    # return [
    #     relay.ones(_.checked_type.shape, dtype=_.checked_type.dtype) for _ in orig.args
    # ]
    """Returns [grad, grad]"""
    # TODO: check why collapse_sum is necessary here
    assert len(orig.args) == 1
    assert isinstance(grad, dict), f"{type(grad)}"

    # print([type(_) for _ in orig.args])
    t = orig.args[0]
    attrs = orig.attrs
    dshape = [int(_) for _ in t.checked_type.shape]
    dtype = t.checked_type.dtype
    # print(dshape, type(dshape))
    # print(attrs.indices_or_sections, attrs.axis)
    indices = [int(_) for _ in attrs.indices_or_sections]
    # print(indices)
    # print(type(indices))
    return_grads = []
    start = 0
    for idx, ind in enumerate(indices):
        if idx in grad:
            return_grads.append(grad[idx])
        else:
            dshape = [int(_) for _ in t.checked_type.shape]
            dshape[attrs.axis] = ind - start
            return_grads.append(relay.zeros(dshape, dtype=dtype))
        start = ind
    idx += 1
    if idx in grad:
        return_grads.append(grad[idx])
    else:
        dshape = [int(_) for _ in t.checked_type.shape]
        dshape[attrs.axis] = dshape[attrs.axis] - start
        return_grads.append(relay.zeros(dshape, dtype=dtype))
    # print(grad[0], len(grad[0]))
    # print(type(return_grads[0]), type(return_grads[1]), type(return_grads[2]))
    # exit(0)
    out_grad = concatenate(return_grads, axis=attrs.axis)
    # [(type(_), _.checked_type.shape, shape_of(_)) for _ in orig.args],
    # print(visualize_call(out_grad))
    return [
        out_grad,
    ]


# TODO: dirty fix for mcu setting
@register_gradient("cast", level=30)
def cast_grad(orig, grad):
    x = orig.args[0]
    return [
        grad,
    ]


# TODO: dirty fix for MCU settings.
@register_gradient("mcumean")
def mcumean_grad(orig, grad):
    """Returns grad broadcasted to data dims"""
    data, axis = orig.args[0], _get_reduce_axis(orig)
    shape = data.checked_type.concrete_shape
    # dtype = data.checked_type.dtype
    dtype = "float32"
    grad, data = [relay.cast(_, dtype) for _ in (grad, data)]

    if axis is None:
        axis = list(range(len(shape)))

    if not orig.attrs.keepdims:
        grad = _unreduce_expand(grad, axis)
    mult = 1.0
    for a in axis:
        mult /= shape[a]
    # print(shape)
    # print(check_call_info(grad))
    # print(axis)
    # rep = [1 for _ in shape]
    return [
        grad
        * relay.const(mult, dtype=dtype)
        * relay.ones_like(data)
        # relay.tile(grad * const(mult, dtype=dtype), reps=1),
        # relay.tile(grad * const(mult, dtype=dtype), reps=1)
    ]
    return [broadcast_to_like(grad * const(mult, dtype=dtype), data)]


@register_gradient("nn.mcutruncate")
def mcutruncate_grad(orig, grad):
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    x = new_inputs[0]
    dtype = "float32"
    # min = orig.attrs.min
    # max = orig.attrs.max
    min = relay.const(orig.attrs.min, dtype=dtype)
    max = relay.const(orig.attrs.max, dtype=dtype)

    mask1 = relay.greater_equal(x, min)
    mask2 = relay.less_equal(x, max)

    # mask = relay.logical_and(mask1, mask2)
    mask = mask1 * mask2
    zeros = relay.zeros_like(grad)
    # mask = relay.cast(mask, "float32")
    return [
        relay.where(mask, grad, zeros),
    ]


@register_gradient("nn.mcuconv2d")
def mcunetconv2d_grad(orig, grad):
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs

    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )
    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
    )
    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            out_channel,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )
    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]
    return [
        backward_data,
        # relay.zeros_like(o_data),
        backward_weight,
        # relay.zeros_like(o_weight),
        backward_bias,
        # relay.zeros_like(o_bias),
        # backward_zero_x,
        relay.zeros_like(o_zx),
        # backward_zero_y,
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_mcunetconv2d_grad_tmp_fix(orig, grad, topk=None):
    from autodiff.diff_ops import (
        broadcast_to_like,
        const,
        get_const_tuple,
        get_pad_tuple,
        _nn,
        tile,
        reshape,
        _sum,
        transpose,
        strided_slice,
    )

    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    # if topk is not None:
    #     tmp_inc = round(topk * in_channel)
    #     tmp_ouc = round(topk * out_channel)
    #     data = relay.strided_slice(data,
    #         begin=relay.const([0, 0, 0, 0]),
    #         end=relay.const([batch, tmp_inc, in_h, in_w]),
    #     )
    #     grad = relay.strided_slice(grad,
    #         begin=relay.const([0, 0, 0, 0]),
    #         end=relay.const([grad_n, tmp_ouc, grad_h, grad_w]),
    #     )

    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, in_channel // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask

    backward_weight = relay.strided_slice(
        backward_weight,
        begin=(0, 0, 0, 0),
        end=(round(topk * out_channel), in_channel, filter_h, filter_w),
    )
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_in_channel_mcunetconv2d_grad(orig, grad, topk=None):
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        assert attrs.groups == 1
        data = relay.strided_slice(
            data,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )

    grad = tile(grad, [1, tmp_inc // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            tmp_inc // attrs.groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, tmp_inc // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_depth_wise_mcunetconv2d_grad(orig, grad, topk=None):
    from autodiff.diff_ops import (
        broadcast_to_like,
        const,
        get_const_tuple,
        get_pad_tuple,
        _nn,
        tile,
        reshape,
        _sum,
        transpose,
        strided_slice,
    )

    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    groups = attrs.groups
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        tmp_ouc = round(topk * out_channel)
        data = relay.strided_slice(
            data,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )
        grad = relay.strided_slice(
            grad,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([grad_n, tmp_ouc, grad_h, grad_w]),
        )
        groups = tmp_inc

    grad = tile(grad, [1, tmp_inc // groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            tmp_inc // groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, tmp_inc // groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)
    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_mcunetconv2d_grad(orig, grad, topk=None):
    from autodiff.diff_ops import (
        broadcast_to_like,
        const,
        get_const_tuple,
        get_pad_tuple,
        _nn,
        tile,
        reshape,
        _sum,
        transpose,
        strided_slice,
    )

    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        tmp_ouc = round(topk * out_channel)
        data = relay.strided_slice(
            data,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )
        grad = relay.strided_slice(
            grad,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([grad_n, tmp_ouc, grad_h, grad_w]),
        )

    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, in_channel // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


@register_gradient("nn.mcuadd")
def mcunetconv2d_grad(orig, grad):
    # cast to 32bits for backward computation
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y = new_inputs
    grad = relay.cast(grad, "float32")

    # grad_zero_y = grad_output.sum([0, 2, 3])
    grad_zero_y = relay.sum(grad)
    # grad_sum = grad_output / scale_y.item()
    new_scale_y = relay.reshape(scale_y, newshape=[1, -1, 1, 1])
    grad_sum = grad / new_scale_y

    # grad_x1 = grad_sum * scale_x1.item()
    new_scale_x1 = relay.reshape(scale_x1, newshape=[1, -1, 1, 1])
    grad_x1 = grad_sum * new_scale_x1

    # grad_x2 = grad_sum * scale_x2.item()
    new_scale_x2 = relay.reshape(scale_x2, newshape=[1, -1, 1, 1])
    grad_x2 = grad_sum * new_scale_x2

    # grad_zero_x1 = - grad_x1.sum([0, 2, 3])
    grad_zero_x1 = -relay.sum(grad_x1)
    # grad_zero_x2 = - grad_x2.sum([0, 2, 3])
    grad_zero_x2 = -relay.sum(grad_x2)
    return [
        grad_x1,
        grad_x2,
        grad_zero_x1,
        grad_zero_x2,
        relay.zeros_like(scale_x1),
        relay.zeros_like(scale_x2),
        grad_zero_y,
        relay.zeros_like(scale_y),
    ]


@register_gradient("nn.log_softmax", level=PROJECT_LEVEL)
def log_softmax_grad(orig, grad):
    """Gradient of log_softmax"""
    return [grad - _sum(grad, axis=orig.attrs.axis, keepdims=True) * exp(orig)]


@register_gradient("nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(orig, grad):
    x, y = orig.args
    # shape = shape_of(x)
    # batch_size = take(shape, const(0, dtype="int32"), axis=0)
    # print(x.checked_type.shape[0], type(x.checked_type.shape[0]))
    batch_size = const(int(x.checked_type.shape[0]))
    # print(batch_size)
    # input()
    grad = grad / batch_size.astype(x.checked_type.dtype)
    return [-grad * y, -grad * x]


# print(GRAD_OP_MAP.keys())
@register_gradient("nn.dense")
def dense_grad(orig, grad):
    x, w = orig.args
    # print("DEBUG", x.checked_type.shape, w.checked_type.shape, grad.checked_type.shape)
    # print("DEBUG dense_grad")
    dydx = relay.nn.matmul(grad, w)
    dydw = relay.nn.matmul(relay.transpose(grad), x)
    return [dydx, dydw]


@register_gradient("nn.bias_add")
def bias_add_grad(orig, grad):
    """Returns gradient of bias_add"""
    data = orig.args[0]
    return [
        # collapse_sum_like(grad, data),
        grad,
        _sum(grad, orig.attrs.axis, keepdims=False, exclude=True),
    ]


@register_gradient("clip")
def clip_grad(orig, grad):
    """Returns grad * (select(x < min || max < x , 0, 1))."""
    x = orig.args[0]
    a_min = orig.attrs.get_int("a_min")
    a_max = orig.attrs.get_int("a_max")
    zeros = zeros_like(x)
    ones = ones_like(x)
    # a_mins = broadcast_to_like(const(a_min, dtype=x.checked_type.dtype), x)
    # a_maxs = broadcast_to_like(const(a_max, dtype=x.checked_type.dtype), x)
    a_mins = relay.zeros(x.checked_type.shape, dtype=x.checked_type.dtype) * const(
        a_min
    )
    a_maxs = relay.ones(x.checked_type.shape, dtype=x.checked_type.dtype) * const(a_max)
    return [where(less(x, a_mins), zeros, where(less(a_maxs, x), zeros, ones * grad))]


@register_gradient("mean")
def mean_grad(orig, grad):
    """Returns grad broadcasted to data dims"""
    data, axis = orig.args[0], _get_reduce_axis(orig)
    shape = data.checked_type.concrete_shape
    if axis is None:
        axis = list(range(len(data.checked_type.concrete_shape)))
    if not orig.attrs.keepdims:
        grad = _unreduce_expand(grad, axis)
    mult = 1.0
    for a in axis:
        mult /= shape[a]
    # return [broadcast_to_like(grad * const(mult, dtype=data.checked_type.dtype), data)]
    return [grad * const(mult, dtype=data.checked_type.dtype)]


@register_gradient("nn.relu")
def relu_grad(orig, grad):
    """Returns grad * (select(x < 0, 0, 1))."""
    x = orig.args[0]
    zeros = relay.zeros_like(x)
    return [
        relay.op.transform.where(relay.less(x, zeros), zeros, grad),
    ]


@register_gradient("add")
def add_grad(orig, grad):
    """Returns [grad, grad]"""
    # TODO: check why collapse_sum is necessary here
    return [grad, grad]


@register_gradient("nn.adaptive_avg_pool2d", level=PROJECT_LEVEL + 1)
def adaptive_avg_pool2d_grad(orig, grad):
    # print(
    #     f"|adaptive_avg_pool2d_grad| (#num of args: {len(orig.args)}):",
    #     [(type(_), _.checked_type.shape, shape_of(_)) for _ in orig.args],
    # )
    """Returns the gradient of adaptive_avg_pool2d."""
    data = orig.args[0]
    shape = data.checked_type.shape
    attrs = orig.attrs  # ['output_size', 'layout', 'out_layout']
    layout = attrs.layout

    output_size = attrs.output_size
    assert layout in ["NCHW", "NHWC"], f"un-supported layout {layout}"
    if layout == "NCHW":
        pool_size = shape[2], shape[3]
    elif layout == "NHWC":
        pool_size = shape[1], shape[2]

    # TODO: fix the shape check
    pool_size = (pool_size[0] // output_size[0], pool_size[1] // output_size[1])

    pool_grad = _nn.avg_pool2d_grad(
        grad, data, pool_size=pool_size, strides=(1, 1), padding=(0, 0), layout=layout
    )
    # print(type(pool_grad), pool_grad)
    return [
        pool_grad,
    ]


@register_gradient("nn.conv2d", level=PROJECT_LEVEL + 1)
def conv2d_grad(orig, grad):
    """Gradient of conv2d"""
    attrs = orig.attrs
    data, weight = orig.args
    data_shape = get_const_tuple(data.checked_type.shape)
    weight_shape = get_const_tuple(weight.checked_type.shape)
    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )
    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(grad_h, grad_w),
        channels=batch * out_channel * in_channel // attrs.groups,
    )
    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            out_channel,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )
    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
        )

    return [backward_data, backward_weight]


# @register_gradient("nn.conv2d", level=PROJECT_LEVEL+10)
# def conv2d_grad(orig, grad):
#     """Gradient of conv2d"""
#     attrs = orig.attrs
#     data, weight = orig.args
#     data_shape = get_const_tuple(data.checked_type.shape)
#     weight_shape = get_const_tuple(weight.checked_type.shape)
#     _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
#     _, _, in_h, in_w = data_shape
#     _, _, filter_h, filter_w = weight_shape

#     # infer output_padding
#     fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
#         get_const_tuple(attrs.padding), (filter_h, filter_w)
#     )
#     stride_h, stride_w = get_const_tuple(attrs.strides)
#     out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
#     out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
#     output_padding = (in_h - out_h, in_w - out_w)

#     assert attrs.data_layout == "NCHW", "only support NCHW data layout"
#     assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
#     assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

#     if attrs.out_dtype in ["", None]:
#         assert data.checked_type, "Call InferType first."
#         out_dtype = data.checked_type.dtype
#     else:
#         out_dtype = attrs.out_dtype

#     backward_data = _nn.conv2d_transpose(
#         grad,
#         weight,
#         strides=attrs.strides,
#         padding=attrs.padding,
#         dilation=attrs.dilation,
#         groups=attrs.groups,
#         output_padding=output_padding,
#         out_dtype=out_dtype,
#     )

#     backward_weight = _nn.conv2d_backward_weight(
#         grad,
#         data,
#         strides=attrs.strides,
#         padding=attrs.padding,
#         dilation=attrs.dilation,
#         groups=attrs.groups,
#         channels=attrs.channels,
#         kernel_size=(filter_h, filter_w),
#         grad_layout=attrs.out_layout if attrs.out_layout else attrs.data_layout,
#         data_layout=attrs.data_layout,
#         kernel_layout=attrs.kernel_layout,
#         out_dtype=out_dtype,
#     )

#     return [backward_data, backward_weight]
