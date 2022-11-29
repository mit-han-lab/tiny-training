import os, os.path as osp
from copy import deepcopy
import json
from textwrap import indent
from types import new_class
from pandas import isna
from sklearn.preprocessing import KernelCenterer

import torch
import torch.nn as nn
import torchvision
from torchvision import models

import numpy as np
from copy import deepcopy

import tvm
from tvm import relay, te
from tvm.contrib import graph_executor

from ..mod import mod_save, mod_load
from ..autodiff.mcuop import *

from .mcunetv3_wrapper import (
    QuantizedConv2dDiff,
    QuantizedMbBlockDiff,
    QuantizedAvgPoolDiff,
    ScaledLinear,
)


def convert_QuantizedConv2dDiff(op_idx, n: QuantizedConv2dDiff, data):
    out, args = mcuconv_factory(
        data,
        prefix=f"{op_idx}_",
        in_channels=n.in_channels,
        out_channels=n.out_channels,
        padding=n.padding,
        strides=n.stride,
        groups=n.groups,
        kernel_size=n.kernel_size,
    )
    tmp_params = extract_mcuconv2d_params(n, args)
    return out, args, tmp_params


def convert_QuantizedMbBlockDiff(op_idx, n: QuantizedMbBlockDiff, data):
    out = data
    sub_n = n
    assert isinstance(sub_n, QuantizedMbBlockDiff)
    assert isinstance(sub_n.conv, nn.Sequential)
    orig_out = out
    tot_params = {}
    tot_args = []
    for idx2, n in enumerate(sub_n.conv):
        assert isinstance(n, QuantizedConv2dDiff)
        # print(f"{op_idx}_{idx2}_", n.bias is not None)
        out, args = mcuconv_factory(
            out,
            prefix=f"{op_idx}_conv_{idx2}_",
            in_channels=n.in_channels,
            out_channels=n.out_channels,
            padding=n.padding,
            strides=n.stride,
            groups=n.groups,
            kernel_size=n.kernel_size,
        )
        tot_args += list(args)
        tmp_params = extract_mcuconv2d_params(n, args)
        tot_params.update(tmp_params)

    # residual
    if sub_n.q_add is not None:
        out, args = mcuadd_factory(
            orig_out, out, out_channels=n.out_channels, prefix=f"{op_idx}_qadd_"
        )
        tot_args += list(args)
        tmp_params = extract_mcuadd_params(sub_n.q_add, args)
        tot_params.update(tmp_params)

    return out, tot_args, tot_params


def convert_ScaledLinear(op_idx, n: ScaledLinear, data):
    out = relay.mcumean(data, axis=[2, 3], keepdims=True)
    # out = relay.nn.mcutruncate(out)
    return out, {}, {}


def convert_QuantizedAvgPoolDiff(op_idx, n: QuantizedAvgPoolDiff, data):
    out = relay.mcumean(data, axis=[2, 3], keepdims=True)
    # out = relay.nn.mcutruncate(out)
    return out, {}, {}


def convert_vww_to_ir(vww_model, input_shape=[1, 3, 80, 80]):
    net = nn.Sequential(
        vww_model[0],
        *vww_model[1],
        *vww_model[2:],
    )
    data = relay.var("input", shape=input_shape, dtype="int8")
    tot_args = [
        data,
    ]
    tot_params = {}
    out = data

    for idx, n in enumerate(net):
        if isinstance(n, nn.Identity):
            continue
        print(idx, type(n))
        if isinstance(n, QuantizedConv2dDiff):
            out, op_args, op_params = convert_QuantizedConv2dDiff(idx, n, out)
        elif isinstance(n, QuantizedMbBlockDiff):
            out, op_args, op_params = convert_QuantizedMbBlockDiff(idx, n, out)
        elif isinstance(n, QuantizedAvgPoolDiff):
            out, op_args, op_params = convert_QuantizedAvgPoolDiff(idx, n, out)
        else:
            raise NotImplementedError
        tot_args += op_args
        tot_params.update(op_params)

    expr = relay.Function(tot_args, out)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    return mod, tot_params, idx


def nn_module_to_ir(model, input_res=[1, 3, 80, 80]):
    fshape = input_res
    net = model
    data = relay.var("input", shape=fshape, dtype="int8")
    tot_args = [
        data,
    ]
    tot_params = {}
    out = data
    if isinstance(net[0], QuantizedConv2dDiff):
        n = net[0]
        out, args = mcuconv_factory(
            data,
            prefix="0_",
            in_channels=n.in_channels,
            out_channels=n.out_channels,
            padding=n.padding,
            strides=n.stride,
            groups=n.groups,
            kernel_size=n.kernel_size,
        )
        tot_args += list(args)

        tmp_params = extract_mcuconv2d_params(net[0], args)
        tot_params.update(tmp_params)

    op_idx = 1
    for sub_n in net[1]:
        assert isinstance(
            sub_n, QuantizedMbBlockDiff
        ), f"get sub_n instance {type(sub_n)}, expect {QuantizedMbBlockDiff}"
        assert isinstance(sub_n.conv, nn.Sequential)
        orig_out = out
        for idx2, n in enumerate(sub_n.conv):
            assert isinstance(n, QuantizedConv2dDiff)
            out, args = mcuconv_factory(
                out,
                prefix=f"{op_idx}_conv_{idx2}_",
                in_channels=n.in_channels,
                out_channels=n.out_channels,
                padding=n.padding,
                strides=n.stride,
                groups=n.groups,
                kernel_size=n.kernel_size,
            )
            tot_args += list(args)
            tmp_params = extract_mcuconv2d_params(n, args)
            tot_params.update(tmp_params)

        # residual
        if sub_n.q_add is not None:
            out, args = mcuadd_factory(
                orig_out, out, out_channels=n.out_channels, prefix=f"{op_idx}_qadd_"
            )
            if op_idx == 11:
                pass
            tot_args += list(args)
            tmp_params = extract_mcuadd_params(sub_n.q_add, args)
            tot_params.update(tmp_params)

        op_idx += 1

    n = net[2]
    if isinstance(n, QuantizedConv2dDiff):
        out, args = mcuconv_factory(
            out,
            prefix=f"{op_idx}_",
            in_channels=n.in_channels,
            out_channels=n.out_channels,
            padding=n.padding,
            strides=n.stride,
            groups=n.groups,
            kernel_size=n.kernel_size,
        )
        tot_args += list(args)
        tmp_params = extract_mcuconv2d_params(n, args)
        tot_params.update(tmp_params)
        op_idx += 1

    out = relay.mcumean(out, axis=[2, 3], keepdims=True)
    assert isinstance(net[-1], QuantizedConv2dDiff), type(net[-1])
    n = net[-1]
    out, args = mcuconv_factory(
        out,
        prefix=f"{op_idx}_",
        in_channels=n.in_channels,
        out_channels=n.out_channels,
        padding=n.padding,
        strides=n.stride,
        groups=n.groups,
        kernel_size=n.kernel_size,
    )
    tot_args += list(args)
    tmp_params = extract_mcuconv2d_params(n, args)
    tot_params.update(tmp_params)

    # for k, v in tot_params.items():
    #     print(k, v.shape, v.numpy().mean(), v.numpy().var())

    # expr = relay.Function(tot_args, out)
    # mod = tvm.IRModule.from_expr(expr)
    # mod = relay.transform.InferType()(mod)

    return out, tot_args, tot_params, op_idx


def nn_seq_to_ir(model, input_res=[1, 3, 80, 80]):
    param_dtype = "int8"
    data = relay.var("input", shape=input_res, dtype=param_dtype)

    param_dtype = "int8"
    tot_args = [
        data,
    ]
    tot_params = {}
    out = data

    from export_samples.pth_vww2json import (
        convert_QuantizedConv2dDiff,
        convert_QuantizedMbBlockDiff,
        convert_vww_to_ir,
        convert_QuantizedAvgPoolDiff,
    )

    for idx, n in enumerate(model):
        if isinstance(n, nn.Identity):
            continue
        # print(idx, type(n))
        if isinstance(n, QuantizedConv2dDiff):
            out, op_args, op_params = convert_QuantizedConv2dDiff(idx, n, out)
        elif isinstance(n, QuantizedMbBlockDiff):
            out, op_args, op_params = convert_QuantizedMbBlockDiff(idx, n, out)
        elif isinstance(n, QuantizedAvgPoolDiff):
            out, op_args, op_params = convert_QuantizedAvgPoolDiff(idx, n, out)
        else:
            raise NotImplementedError(f"{idx}: {type(n)}, not suportted")
        tot_args += op_args
        t_op = {}
        for k, v in op_params.items():
            # if k[0].isdigit():
            #     k = 'v' + k
            t_op[k] = v
        tot_params.update(t_op)

    expr = relay.Function(tot_args, out)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    return out, tot_args, tot_params, idx


if __name__ == "__main__":
    from compilation.convert.load_mcunetv3 import build_quantized_mcunet

    net, rs = build_quantized_mcunet()
    batch_size = 1
    fshape = (batch_size, 3, rs, rs)
    fshape_str = f"{fshape[0]}x{fshape[1]}x{fshape[2]}x{fshape[3]}"
    path = ".tmp/vww-full"

    mod, tot_params, op_idx = convert_vww_to_ir(net)

    # from .ir2json import translate_ir
    # mod_save(mod, tot_params, path=f"{path}", mod_name=f"fwd-{fshape_str}.ir")
    # print(f"{path}/fwd-{fshape_str}.ir")
    # translate_ir(path=f"{path}/fwd-{fshape_str}.ir", out_folder=".model/vww")
