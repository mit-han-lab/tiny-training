import time
from turtle import backward

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
    tile,
)
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple

from .op2grad import register_gradient, GRAD_OP_MAP

# from graph_tools.visualize_call import visualize_call, check_call_dtype, check_call_shape


def check_call_dtype(call):
    vars = relay.analysis.all_vars(call)
    expr = relay.Function(vars, call)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    expr = mod["main"]
    return expr.body.checked_type.dtype


def check_call_shape(call):
    vars = relay.analysis.all_vars(call)
    expr = relay.Function(vars, call)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    expr = mod["main"]
    return expr.body.checked_type.shape


def visualize_call(call, infer=False):
    vars = relay.analysis.all_vars(call)
    expr = relay.Function(vars, call)
    if infer:
        mod = tvm.IRModule.from_expr(expr)
        mod = relay.transform.InferType()(mod)
        expr = mod["main"]
    return expr


def get_weight_scales(w, n_bits=8, axis=1):
    wmax = relay.max(relay.abs(w), axis=axis, keepdims=True)
    dtype = check_call_dtype(wmax)
    return wmax / relay.const(2 ** (n_bits - 1) - 1, dtype=dtype)


@register_gradient("nn.mcuadd")
def mcuadd_int8_grad(orig, grad):
    # cast to 32bits for backward computation
    # new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    new_inputs = orig.args
    x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y = new_inputs
    ograd = grad
    grad_dtype = check_call_dtype(grad)

    grad = relay.cast(grad, "float32")

    grad_zero_y = relay.sum(grad)
    new_scale_y = relay.reshape(scale_y, newshape=[1, -1, 1, 1])
    grad_sum = grad / new_scale_y

    new_scale_x1 = relay.reshape(scale_x1, newshape=[1, -1, 1, 1])
    grad_x1 = grad_sum * new_scale_x1

    new_scale_x2 = relay.reshape(scale_x2, newshape=[1, -1, 1, 1])
    grad_x2 = grad_sum * new_scale_x2

    grad_zero_x1 = -relay.sum(grad_x1)
    grad_zero_x2 = -relay.sum(grad_x2)

    # print(grad_dtype, check_call_shape(grad_x1), check_call_shape(grad_x2))
    # input()
    return [
        relay.cast(grad_x1, grad_dtype),
        relay.cast(grad_x2, grad_dtype),
        # ograd, ograd,
        grad_zero_x1,
        grad_zero_x2,
        relay.zeros_like(scale_x1),
        relay.zeros_like(scale_x2),
        grad_zero_y,
        relay.zeros_like(scale_y),
    ]


@register_gradient("nn.mcutruncate")
def mcutruncate_int8_grad(orig, grad):
    new_inputs = orig.args
    x = new_inputs[0]
    dtype = check_call_dtype(x)
    min = relay.const(orig.attrs.min, dtype=dtype)
    max = relay.const(orig.attrs.max, dtype=dtype)

    mask1 = relay.greater_equal(x, min)
    mask2 = relay.less_equal(x, max)

    mask = mask1 * mask2
    zeros = relay.zeros_like(grad)
    return [
        relay.where(mask, grad, zeros),
        # grad
    ]


def post_process_gradients(backward_data, backward_weight, eps=None):
    w_scales = get_weight_scales(backward_weight, n_bits=8)
    x_scales = get_weight_scales(backward_data, n_bits=8)
    # x_scales = relay.cast(x_scales, "float32")
    if eps is None:
        backward_data = backward_data / x_scales
        backward_weight = backward_weight / w_scales
    else:
        backward_data = relay.cast(backward_data, "float32")
        backward_weight = relay.cast(backward_weight, "float32")
        x_scales = relay.cast(x_scales, "float32")
        w_scales = relay.cast(w_scales, "float32")
        backward_data = backward_data / (x_scales + relay.const(1e-12, dtype="float32"))
        backward_weight = backward_weight / (
            w_scales + relay.const(1e-12, dtype="float32")
        )
    backward_data = relay.cast(backward_data, dtype="int8")
    backward_weight = relay.cast(backward_weight, dtype="int8")
    return backward_data, backward_weight


@register_gradient("nn.mcuconv2d")
def mcunetconv2d_int8_grad(orig, grad):
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)
    data_dtype = o_data.checked_type.dtype
    weight_dtype = o_weight.checked_type.dtype

    # cast to int32 during backward computation
    ograd = grad
    # new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    # grad = relay.cast(grad, "float32")
    new_inputs = orig.args
    data, weight, bias, zx, zy, scale = orig.args

    # scale = relay.reshape(scale, newshape=[1, -1, 1, 1])
    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    # grad = grad * scale
    dtype = "float32"
    out_dtype = "float32"
    if "int" in str(weight_dtype) and "int" in str(data_dtype):
        out_dtype = "int32"
    # print(data_dtype, weight_dtype )
    tmp_grad = relay.cast(grad, dtype=out_dtype)
    backward_bias = relay.sum(tmp_grad, axis=1, exclude=True)
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

    grad_dtype = check_call_dtype(grad)
    conv_out_dtype = "float32"
    temp_weight = weight
    if grad_dtype != weight_dtype:
        temp_weight = relay.cast(weight, grad_dtype)
    temp_weight_dtype = check_call_dtype(temp_weight)
    if "int" in str(grad_dtype) and "int" in str(temp_weight_dtype):
        conv_out_dtype = "int32"
    # print(check_call_dtype(ograd), check_call_dtype(o_weight))
    # print(grad_dtype, temp_weight_dtype)
    # input()
    backward_data = _nn.conv2d_transpose(
        grad,
        temp_weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
        out_dtype=conv_out_dtype,
    )
    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    conv_out_dtype = "float32"
    temp_data = data
    if data_dtype != grad_dtype:
        temp_data = relay.cast(data, grad_dtype)
    temp_data_dtype = check_call_dtype(temp_data)
    if "int" in str(temp_data_dtype) and "int" in str(grad_dtype):
        conv_out_dtype = "int32"
    backward_weight = _nn.conv2d(
        temp_data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
        out_dtype=conv_out_dtype,
    )
    # print(check_call_shape(data), check_call_shape(grad), check_call_shape(backward_weight))
    # print(check_call_dtype(temp_data), check_call_dtype(grad), check_call_dtype(backward_weight))
    # input()
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
    backward_data, backward_weight = post_process_gradients(
        backward_data, backward_weight
    )

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_in_channel_mcunetconv2d_int8grad(orig, grad, topk=None):
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
    data_dtype = o_data.checked_type.dtype
    weight_dtype = o_weight.checked_type.dtype

    # cast to int32 during backward computation
    ograd = grad
    # new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    # grad = relay.cast(grad, "float32")
    new_inputs = orig.args
    data, weight, bias, zx, zy, scale = orig.args

    # scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    # grad = grad * scale
    dtype = "float32"
    out_dtype = "float32"
    if "int" in str(weight_dtype) and "int" in str(data_dtype):
        out_dtype = "int32"
    # print(data_dtype, weight_dtype )
    tmp_grad = relay.cast(grad, dtype=out_dtype)
    backward_bias = relay.sum(tmp_grad, axis=1, exclude=True)
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

    grad_dtype = check_call_dtype(grad)
    conv_out_dtype = "float32"

    temp_weight = weight
    if grad_dtype != weight_dtype:
        temp_weight = relay.cast(weight, grad_dtype)
    temp_weight_type = check_call_dtype(temp_weight)
    if "int" in str(grad_dtype) and "int" in str(temp_weight_type):
        conv_out_dtype = "int32"
    backward_data = _nn.conv2d_transpose(
        grad,
        temp_weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
        out_dtype=conv_out_dtype,
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

    conv_out_dtype = "float32"
    if "int" in str(data_dtype) and "int" in str(grad_dtype):
        conv_out_dtype = "int32"
    temp_data = data
    if data_dtype != grad_dtype:
        temp_data = relay.cast(data, grad_dtype)
    backward_weight = _nn.conv2d(
        temp_data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
        out_dtype=conv_out_dtype,
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
    #
    backward_data, backward_weight = post_process_gradients(
        backward_data, backward_weight
    )

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_depth_wise_mcunetconv2d_int8grad(orig, grad, topk=None):
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
    data_dtype = o_data.checked_type.dtype
    weight_dtype = o_weight.checked_type.dtype

    # cast to int32 during backward computation
    ograd = grad
    # new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    # grad = relay.cast(grad, "float32")
    new_inputs = orig.args
    data, weight, bias, zx, zy, scale = orig.args

    # scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    # grad = grad * scale
    dtype = "float32"
    out_dtype = "float32"
    if "int" in str(weight_dtype) and "int" in str(data_dtype):
        out_dtype = "int32"
    # print(data_dtype, weight_dtype )
    tmp_grad = relay.cast(grad, dtype=out_dtype)
    backward_bias = relay.sum(tmp_grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs

    # _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
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

    grad_dtype = check_call_dtype(grad)
    conv_out_dtype = "float32"
    if "int" in str(grad_dtype) and "int" in str(weight_dtype):
        conv_out_dtype = "int32"
    temp_weight = weight
    if grad_dtype != weight_dtype:
        temp_weight = relay.cast(weight, grad_dtype)
    backward_data = _nn.conv2d_transpose(
        grad,
        temp_weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        # to fix codegen bug
        # TODO(lyken17): figure out why missing default value leads to error
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
        out_dtype=conv_out_dtype,
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

    conv_out_dtype = "float32"
    if "int" in str(data_dtype) and "int" in str(grad_dtype):
        conv_out_dtype = "int32"
    temp_data = data
    if data_dtype != grad_dtype:
        temp_data = relay.cast(data, grad_dtype)
    backward_weight = _nn.conv2d(
        temp_data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
        out_dtype=conv_out_dtype,
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
    #
    backward_data, backward_weight = post_process_gradients(
        backward_data, backward_weight
    )

    # TODO: update truncation mask
    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]
