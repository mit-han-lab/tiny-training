import tvm
from tvm import relay, te
import torch
from tvm.relay.frontend.pytorch import (
    Prelude,
    PyTorchOpConverter,
    _run_jit_passes,
    get_all_op_names,
    _get_relay_input_vars,
    convert_params,
    _get_input_names,
    _get_operator_nodes,
    _analysis,
)


def convert_ir_var(temp):
    if isinstance(temp, tvm.ir.container.Array):
        return list([convert_ir_var(_) for _ in temp])
    elif isinstance(temp, tvm.tir.expr.IntImm):
        return int(temp)
    elif isinstance(temp, tvm.runtime.container.String):
        return str(temp)
    elif isinstance(temp, (float, int, str)):
        return temp
    elif temp is None:
        return temp
    if isinstance(temp, (tuple, list)):
        return list(convert_ir_var(_) for _ in temp)
    else:
        raise NotImplementedError(type(temp), temp, isinstance(temp, int))


# def compute_gradient(mod, requires_grad=None):
#     bwd_mod = relay.transform.gradient(mod["main"], mode="first_order")
#     new_mod = tvm.IRModule.from_expr(bwd_mod)
#     new_mod = relay.transform.InferType()(new_mod)
#     fmod = tvm.transform.Sequential([
#         relay.transform.ToGraphNormalForm(),
#         relay.transform.FoldConstant(),
#         relay.transform.DeadCodeElimination()
#     ])(new_mod)

#     if requires_grad is not None:
#         vs = relay.analysis.all_vars(fmod['main'])
#         out_node = fmod['main'].body[0]
#         grads = fmod['main'].body[1]
#         gs = []
#         print("All args: ", [_.name_hint for _ in vs])
#         print("Require grad args: ", requires_grad)
#         for v, g in zip(vs, grads):
#             if str(v.name_hint) in requires_grad:
#                 gs.append(g)
#         fn = tvm.relay.Function(vs, relay.Tuple([out_node, *gs]))
#         fmod = tvm.IRModule.from_expr(fn)

#     return fmod

from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor
from tvm.relay.expr_functor import ExprMutator, Call


class ExprTranform(ExprMutator):
    def __call__(self, expr):
        if isinstance(expr, relay.expr.Expr):
            mod = tvm.IRModule.from_expr(expr)
            mod = relay.transform.InferType()(mod)
        elif isinstance(expr, tvm.IRModule):
            pass
        else:
            raise NotImplementedError(type(expr))

        mod = relay.transform.InferType()(mod)
        expr = mod["main"]
        return self.visit(expr)


class SimplifyReshape(ExprTranform):
    def visit_call(self, call):
        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
        if str(call.op) == "reshape":
            target_shape = call.attrs.newshape
            current_shape = args[0].checked_type.shape
            output_shape = call.checked_type.shape

            if len(output_shape) == len(current_shape) and all(
                [a == b for a, b in zip(output_shape, current_shape)]
            ):
                return args[0]
        return Call(new_fn, args, call.attrs)





class ChangeDataType(ExprMutator):
    def __init__(self, dtype_map={"float32": "int32"}):
        super().__init__()
        self.dtype_map = dtype_map

    def visit_var(self, var):
        # print(var.name_hint)
        if var.type_annotation.dtype in self.dtype_map:
            print(
                f"[{var.name_hint}] Change data type from {var.type_annotation.dtype} to {self.dtype_map[var.type_annotation.dtype]}"
            )
            d = var.type_annotation.shape
            var_new = relay.var(
                var.name_hint, shape=d, dtype=self.dtype_map[var.type_annotation.dtype]
            )
            return var_new
        else:
            # TODO: figure out why this failed.
            # super(self).visit_var(var)
            return var

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
        return Call(new_fn, args, call.attrs)


from tvm.relay.op import zeros_like, ones_like, less, greater
from tvm.relay.op.transform import where


def truncuate(x, th=127):
    zeros = zeros_like(x)
    threashold = ones_like(x) * relay.const(127)
    return where(greater(x, threashold), threashold, x)


class TruncateOP(ExprMutator):
    def __init__(self, truncuate_ops=["nn.conv2d", "nn.dense"]):
        super().__init__()
        self.truncuate_ops = truncuate_ops

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))

        new_call = Call(new_fn, args, call.attrs)
        if str(call.op) in self.truncuate_ops:
            print(f"truncuate [{call.op}]")
            new_call = truncuate(new_call)
        return new_call


def parse_mod(path="tmp/mod.ir", dtype="int32"):
    with open(path, "r") as fp:
        code = fp.read()
    # code = code.replace("float32", dtype)
    mod_expr = tvm.parser.parse_expr(code)
    int8_mod = tvm.IRModule.from_expr(mod_expr)
    int8_mod = relay.transform.InferType()(int8_mod)
    return int8_mod


def from_pytorch(
    script_module,
    input_infos,
    custom_convert_map=None,
    default_dtype="float32",
    use_parser_friendly_name=False,
    keep_quantized_weight=False,
    return_raw_output=False,
):
    mod = tvm.IRModule()
    prelude = Prelude(mod)

    converter = PyTorchOpConverter(prelude, default_dtype)
    # add more hook
    converter.convert_map["aten::tile"] = converter.repeat

    graph = script_module.graph.copy()
    _run_jit_passes(graph)

    if custom_convert_map:
        converter.update_convert_map(custom_convert_map)

    op_names = get_all_op_names(graph)
    converter.report_missing_conversion(op_names)

    is_module = isinstance(script_module, torch.jit.ScriptModule)
    params = script_module.state_dict() if is_module else {}
    outputs = _get_relay_input_vars(
        graph, input_infos, prelude, default_dtype=default_dtype, is_module=is_module
    )

    if use_parser_friendly_name:
        new_names = [key.replace(".", "_") for key in params.keys()]
        params = dict(zip(new_names, params.values()))

    param_vars, tensors, packed_param_map = convert_params(
        graph, params, use_parser_friendly_name
    )

    tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}

    outputs.update(param_vars)
    ret_name = _get_input_names(graph.return_node())

    # For quantized models
    quantized_ops = set(["aten::quantize_per_tensor", "quantized::linear_dynamic"])
    if len(quantized_ops.intersection(set(op_names))) > 0:
        weight_quant_params = qnn_torch.get_weight_quant_params(
            script_module, packed_param_map.values()
        )
        input_scales_for_bias = qnn_torch.add_input_quant_params_to_op_inputs(graph)
        qnn_torch.add_quant_params_to_outputs(
            outputs,
            packed_param_map,
            weight_quant_params,
            input_scales_for_bias,
            keep_quantized_weight,
        )
        qnn_torch.add_quant_params(tvm_params, weight_quant_params)
        converter.update_convert_map(qnn_torch.convert_map)

    ret = converter.convert_operators(
        _get_operator_nodes(graph.nodes()), outputs, ret_name
    )[0]
    if isinstance(ret, list):
        from tvm.relay import expr as _expr

        # ListConstruct kept original python list. Convert to tuple.
        ret = _expr.Tuple(ret)

    # Separate data inputs and parameters to make sure data inputs come first.
    func_args = []
    data_inputs = []
    for arg in _analysis.free_vars(ret):
        if arg.name_hint not in tvm_params.keys():
            data_inputs.append(arg)
        else:
            func_args.append(arg)
    full_args = data_inputs + func_args
    if return_raw_output:
        return (data_inputs, func_args), ret, tvm_params

    mod["main"] = tvm.relay.Function(full_args, ret)
    return relay.transform.RemoveUnusedFunctions()(mod), tvm_params


if __name__ == "__main__":
    scripted_model = torch.jit.trace(model, input_data).eval()
    (data_inputs, func_args), ret, params = from_pytorch(
        scripted_model,
        shape_list,
        use_parser_friendly_name=True,
        return_raw_output=True,
    )
