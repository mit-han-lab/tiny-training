from site import venv

import numpy as np

import tvm
from tvm import relay
from tvm.relay.op import reg
from tvm.relay.op.nn import _make


def truncate_max(x: relay.expr.Call, th=127, dtype="int8"):
    zeros = relay.zeros_like(x)
    threashold = relay.ones_like(x) * relay.const(th, dtype=dtype)
    # return relay.greater(x, threashold)
    return relay.where(relay.greater(x, threashold), threashold, x)


def truncate_min(x: relay.expr.Call, th=-128, dtype="int8"):
    threashold = relay.ones_like(x) * relay.const(th, dtype=dtype)
    return relay.where(relay.less(x, threashold), threashold, x)


@reg.register_legalize("mcumean", level=30)
def mcumean_calc(attrs, inputs, types):
    x = inputs[0]
    dtype = "float32"
    nx = relay.cast(x, dtype)
    out = relay.mean(nx, axis=attrs.axis, keepdims=attrs.keepdims)
    # out = truncuate_max(out, dtype=dtype)
    # out = truncuate_min(out, dtype=dtype)
    out = relay.round(out)
    return relay.cast(out, types[-1].dtype)


@reg.register_legalize("nn.mcutruncate", level=30)
def mcu_nn_truncate(attrs, inputs, types):
    x = inputs[0]
    dtype = types[0].dtype
    int8_res = truncate_max(x, th=127, dtype=dtype)
    int8_res = truncate_min(int8_res, th=-128, dtype=dtype)
    return relay.cast(int8_res, types[1].dtype)


@reg.register_legalize("nn.mcuadd", level=30)
def mcu_nn_add(attrs, inputs, types):
    new_inputs = [relay.cast(_, "float32") for _ in inputs]
    x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y = new_inputs

    scale_x1 = relay.reshape(scale_x1, newshape=[1, -1, 1, 1])
    scale_x2 = relay.reshape(scale_x2, newshape=[1, -1, 1, 1])
    scale_y = relay.reshape(scale_y, newshape=[1, -1, 1, 1])

    x1 = (x1 - zero_x1) * scale_x1
    x2 = (x2 - zero_x2) * scale_x2
    out = x1 + x2
    int32_res = relay.round(out / scale_y)
    int32_res = int32_res + zero_y
    return relay.cast(int32_res, types[-1].dtype)
    # int32_res = truncuate_max(int32_res)
    # int32_res = truncuate_min(int32_res)
    # return relay.cast(int32_res, "int8")


@reg.register_legalize("nn.mcuconv2d", level=20)
def mcu_nn_conv2d(attrs, inputs, types):
    new_inputs = [relay.cast(_, "float32") for _ in inputs]
    x, weight, bias, zx, zy, scale = new_inputs
    # scale = new_inputs[-1]
    nx = x - zx
    conv_res = relay.nn.conv2d(
        nx, weight, attrs.strides, attrs.padding, attrs.dilation, attrs.groups
    )
    conv_res = relay.nn.bias_add(conv_res, bias)

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])
    conv_res = conv_res * scale

    # int32_out = conv_res + relay.cast(zy, "float32")
    # int32_out = relay.round(int32_out)
    int32_out = conv_res
    int32_out = relay.round(int32_out)
    int32_out = int32_out + relay.cast(zy, "float32")
    # return relay.cast(int32_out, types[-1].dtype)
    # int32_out = truncuate_max(int32_out, th=127, dtype="float32")
    # int32_out = truncuate_min(int32_out, th=-128, dtype="float32")
    # return relay.nn.mcutruncate(int32_out)
    return relay.cast(int32_out, types[-1].dtype)


def mcuconv_factory(
    features,
    prefix="",
    in_channels=3,
    out_channels=3,
    kernel_size=3,
    strides=1,
    padding=1,
    groups=1,
    param_dtype="int8",
):
    if isinstance(kernel_size, (list, tuple)):
        ks = kernel_size[0]
    else:
        ks = kernel_size
    weight = relay.var(
        f"{prefix}weight",
        shape=[out_channels, in_channels // groups, ks, ks],
        dtype=param_dtype,
    )
    bias = relay.var(
        f"{prefix}bias",
        shape=[
            out_channels,
        ],
        dtype="int32",
    )
    zero_x = relay.var(
        f"{prefix}zero_x",
        shape=[
            1,
        ],
        dtype=param_dtype,
    )
    zero_y = relay.var(
        f"{prefix}zero_y",
        shape=[
            1,
        ],
        dtype=param_dtype,
    )
    scale = relay.var(
        f"{prefix}scale",
        shape=[
            out_channels,
        ],
        dtype="float32",
    )

    out = relay.nn.mcuconv2d(
        features,
        weight,
        bias,
        zero_x,
        zero_y,
        scale,
        strides=strides,
        padding=padding,
        groups=groups,
    )

    return (
        relay.nn.mcutruncate(out),
        # out,
        (weight, bias, zero_x, zero_y, scale),
    )


def mcuadd_factory(num1, num2, out_channels=3, prefix="", param_dtype="int8"):
    # weight = relay.var(f"{prefix}weight", shape=[out_channels, in_channels], dtype="int8")
    zero_x1 = relay.var(
        f"{prefix}zero_x1",
        shape=[
            1,
        ],
        dtype=param_dtype,
    )
    zero_x2 = relay.var(
        f"{prefix}zero_x2",
        shape=[
            1,
        ],
        dtype=param_dtype,
    )
    zero_y = relay.var(
        f"{prefix}zero_y",
        shape=[
            1,
        ],
        dtype=param_dtype,
    )

    scale_x1 = relay.var(
        f"{prefix}scale_x1",
        shape=[
            1,
        ],
        dtype="float32",
    )
    scale_x2 = relay.var(
        f"{prefix}scale_x2",
        shape=[
            1,
        ],
        dtype="float32",
    )
    scale_y = relay.var(
        f"{prefix}scale_y",
        shape=[
            1,
        ],
        dtype="float32",
    )

    out = relay.nn.mcuadd(
        num1, num2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y
    )
    return (
        relay.nn.mcutruncate(out),
        (zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y),
    )


def extract_mcuconv2d_params(module, args, param_dtype="int8"):
    params = {}
    # weight
    vname = args[0].name_hint
    vtensor = module.weight.detach().numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    # bias
    vname = args[1].name_hint
    vtensor = module.bias.detach().numpy().astype("int32")
    params[vname] = tvm.nd.array(vtensor)

    # 0_zero_x
    vname = args[2].name_hint
    vtensor = module.zero_x.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    # 0_zero_y
    vname = args[3].name_hint
    vtensor = module.zero_y.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    # effective_scale
    vname = args[4].name_hint
    vtensor = module.effective_scale.detach().numpy().astype("float32")
    params[vname] = tvm.nd.array(vtensor)

    vs = vname.split("_")[:-1]
    if hasattr(module, "x_scale"):
        # print(f"{vname} HAS x_scale")
        vname = "_".join(
            vs
            + [
                "x_scale",
            ]
        )
        vtensor = np.array(module.x_scale).astype("float32")
        params[vname] = tvm.nd.array(vtensor)

    if hasattr(module, "y_scale"):
        # print(f"{vname} HAS y_scale")
        vname = "_".join(
            vs
            + [
                "y_scale",
            ]
        )
        vtensor = np.array(module.y_scale).astype("float32")
        params[vname] = tvm.nd.array(vtensor)

    return params


def extract_mcuadd_params(module, args, param_dtype="int8"):
    params = {}

    vname = args[0].name_hint
    vtensor = module.zero_x1.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    vname = args[1].name_hint
    vtensor = module.zero_x2.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    vname = args[2].name_hint
    vtensor = module.scale_x1.detach().numpy().astype("float32")
    params[vname] = tvm.nd.array(vtensor)

    vname = args[3].name_hint
    vtensor = module.scale_x2.detach().numpy().astype("float32")
    params[vname] = tvm.nd.array(vtensor)

    vname = args[4].name_hint
    vtensor = module.zero_y.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    vname = args[5].name_hint
    vtensor = module.scale_y.detach().numpy().astype("float32")
    params[vname] = tvm.nd.array(vtensor)

    return params
