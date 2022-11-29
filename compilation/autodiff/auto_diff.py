import enum
import time
from pprint import pprint
from dask import visualize

import numpy as np

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# from graph_tools.visualize_call import visualize_call

import tvm
from tvm import relay
from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor, Call

from .diff_ops import (
    GRAD_OP_MAP,
    sparse_depth_wise_mcunetconv2d_grad,
    sparse_in_channel_mcunetconv2d_grad,
    sparse_mcunetconv2d_grad_tmp_fix,
)


def bias_only(v):
    if "_bias" in v.name_hint:
        return True
    return False


def full_bp(v):
    return True


def dummy_print(*args, **kwargs):
    return


def real_print(*args, **kwargs):
    return print(*args, **kwargs)


def appending_loss(mod, params, name="label", label_shape=[1, 10]):
    data_inputs = []
    func_args = []
    for v in relay.analysis.all_vars(mod["main"]):
        if v.name_hint in params:
            func_args.append(v)
        else:
            data_inputs.append(v)

    ret = mod["main"].body
    # add label and loss to the DAG
    label = relay.var(name, shape=label_shape)
    zz = relay.nn.log_softmax(ret)
    loss = relay.nn.cross_entropy_with_logits(zz, label)

    new_args = (
        data_inputs
        + [
            label,
        ]
        + func_args
    )
    func = tvm.relay.Function(new_args, loss)
    mod = tvm.IRModule.from_expr(func)

    names = [_.name_hint for _ in new_args]
    return mod, params, names


from tvm.relay.expr import Tuple, TupleGetItem


class AutoDiff(ExprVisitor):
    def __init__(
        self,
        debug=True,
        sparse_op_idx={},
    ):
        super().__init__()
        self.names = set()
        self.grads = dict()
        self.var_grads = dict()
        self.last_call = None
        self.debug = debug
        # TODO: this is a dirty fix
        self.op_idx = 1
        self.sparse_op_idx = sparse_op_idx
        self.sparse_update_meta_info = []
        self.int8_grad = False

    def compute_grad(self, expr):
        self.vars = relay.analysis.all_vars(expr)
        self.names = set()
        self.grads = dict()
        self.var_grads = dict()
        self.last_call = None
        self.visit(expr)

    def visit_tuple(self, tup):
        try:
            tuple_gradients = self.grads[hex(tup.handle.value)][0]
        except KeyError:
            if len(self.grads.keys()) != 0:
                raise
            gs = []
            for idx, field in enumerate(tup.fields):
                # dtype = check_call_dtype(field)
                # shape = check_call_shape(field)
                gs.append(relay.ones_like(field))
            tuple_gradients = relay.Tuple(gs)
            self.grads[hex(tup.handle.value)] = tuple_gradients

        for idx, field in enumerate(tup.fields):
            grad = relay.TupleGetItem(tuple_gradients, idx)
            self.grads[hex(field.handle.value)] = (grad, "TupleVisit")
            if isinstance(field, relay.expr.Var):
                self.var_grads[field.name_hint] = (grad, "TupleVisit")
        return Tuple([self.visit(field) for field in tup.fields], tup.span)

    def visit_tuple_getitem(self, op):
        """
        input: op.tuple_value
        args: op.index
        output: op
        """
        item_gradients = self.grads[hex(op.handle.value)][0]
        arg = op.tuple_value
        addr = hex(arg.handle.value)
        if addr not in self.grads:
            self.grads[addr] = ({op.index: item_gradients}, "TupleGetItem")
        else:
            self.grads[addr][0][op.index] = item_gradients
        if isinstance(arg, relay.expr.Var):
            self.var_grads[arg.name_hint] = (item_gradients, "PlaceHolder")

        # recursively parse the AST
        tuple_value = self.visit(op.tuple_value)
        if not tuple_value.same_as(op.tuple_value):
            return TupleGetItem(tuple_value, op.index)
        return op

    def visit_call(self, call: relay.expr.Call):
        if self.debug:
            dprint = real_print
        else:
            dprint = dummy_print
        call_op = str(call.op)
        assert (
            call_op != "nn.batch_norm"
        ), "batch norm is not supported yet, please fuse the BN"
        dprint("==" * 40)
        addr = hex(call.handle.value)

        # dprint("OP:", call.op, call.args[0].checked_type.shape, "=>", call.checked_type.shape)
        dprint(
            f"[AutoDiff][{call.op}] #num of args:",
            len(call.args),
            [type(_) for _ in call.args],
        )
        if addr not in self.grads:
            grad_output = relay.ones_like(call)
        else:
            grad_output, name_hint = self.grads[addr]

        if call_op not in GRAD_OP_MAP:
            raise NotImplementedError(
                f"[AutoDiff] |{call.op}| not registered in GRAD_OP_MAP"
            )
        else:
            grad_fn = GRAD_OP_MAP[call_op]
            is_sparse = False
            if call_op != "nn.mcuconv2d":
                gs = grad_fn(call, grad_output)
            else:
                from compilation.utils import convert_ir_var

                attrs = call.attrs
                from tvm.topi.utils import get_const_tuple

                if self.op_idx in self.sparse_op_idx:
                    from .diff_ops import sparse_mcunetconv2d_grad
                    from .int8_grad import (
                        sparse_depth_wise_mcunetconv2d_int8grad,
                        sparse_in_channel_mcunetconv2d_int8grad,
                    )

                    ks = call.args[1].checked_type.shape[-1]
                    data, weight, *_ = call.args
                    data_shape = get_const_tuple(data.checked_type.shape)
                    weight_shape = get_const_tuple(weight.checked_type.shape)
                    if self.int8_grad:
                        in_chanel_sparse_bp = sparse_in_channel_mcunetconv2d_int8grad
                        depthwise_sparse_bp = sparse_depth_wise_mcunetconv2d_int8grad
                    else:
                        in_chanel_sparse_bp = sparse_in_channel_mcunetconv2d_grad
                        depthwise_sparse_bp = sparse_depth_wise_mcunetconv2d_grad
                    if self.sparse_op_idx[self.op_idx] < 1:
                        if ks == 1:
                            print(
                                f"[point-wise][int8: {self.int8_grad}] Special handlding for sparse bp nn.mcuconv2d",
                                self.op_idx,
                                call.args[1].checked_type.shape,
                                self.sparse_op_idx[self.op_idx],
                            )
                            gs = in_chanel_sparse_bp(
                                call, grad_output, topk=self.sparse_op_idx[self.op_idx]
                            )
                        elif (
                            attrs.groups == data_shape[1]
                            and data_shape[1] == weight_shape[0]
                        ):
                            print(
                                f"[depth-wise][int8: {self.int8_grad}] Special handlding for sparse bp nn.mcuconv2d",
                                self.op_idx,
                                call.args[1].checked_type.shape,
                                self.sparse_op_idx[self.op_idx],
                            )
                            gs = depthwise_sparse_bp(
                                call, grad_output, topk=self.sparse_op_idx[self.op_idx]
                            )
                        else:
                            raise NotImplementedError(
                                f"ks={ks}, {attrs.groups}, {data_shape[1]}, {weight_shape[0]}"
                            )
                    else:
                        print(
                            f"[full-update {ks}x{ks}][int8: {self.int8_grad}] Special handlding for sparse bp nn.mcuconv2d",
                            self.op_idx,
                            call.args[1].checked_type.shape,
                            self.sparse_op_idx[self.op_idx],
                        )
                        gs = grad_fn(call, grad_output)
                    self.sparse_update_meta_info.append(
                        {
                            "op_idx(revser order)": self.op_idx,
                            "sparse ratio": self.sparse_op_idx[self.op_idx],
                            "gradient shape": convert_ir_var(
                                call.args[1].checked_type.shape
                            ),
                        }
                    )
                    is_sparse = True
                    # gs = sparse_mcunetconv2d_grad(call, grad_output, topk=self.sparse_op_idx[self.op_idx])
                else:
                    gs = grad_fn(call, grad_output)
                print(
                    "OP ",
                    self.op_idx,
                    call.args[0].checked_type.shape,
                    "=>",
                    call.checked_type.shape,
                )
                self.op_idx += 1

            # assign gradients to each input args
            assert len(call.args) == len(
                gs
            ), f"    {call.op} | args: {len(call.args)}, gradients: {len(gs)}"
            for arg, grad in zip(call.args, gs):
                # type / shape checking
                # mod = tvm.IRModule.from_expr(grad)
                # mod = relay.transform.InferType()(mod)
                # print(mod['main'].body.checked_type.shape)
                self.grads[hex(arg.handle.value)] = (grad, str(call_op))
                if isinstance(arg, relay.expr.Var):
                    # dprint(arg.name_hint)
                    self.var_grads[arg.name_hint] = (grad, is_sparse)

        # recursively parse the AST
        new_fn = self.visit(call.op)
        for a in list(call.args)[::-1]:
            self.visit(a)
        return call

        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args[::-1]]
        return new_fn

    def obtain_grads(
        self,
        filter_fn=lambda x: True,
        reverse=True,
    ):
        names = []
        needed_gradients = []
        print("Obtaining gradients")
        # time.sleep(2)
        for v in self.vars:
            if v.name_hint in self.var_grads and filter_fn(
                v, self.var_grads[v.name_hint]
            ):
                names.append(v.name_hint)
                needed_gradients.append(self.var_grads[v.name_hint][0])
        return names[::-1], needed_gradients[::-1]


def compute_autodiff(
    mod,
    keep_prediction=True,
    filter_fn=lambda x, g: True,
    sparse_op_idx={},
    return_sparse_meta_info=False,
    return_gradient_tensors=False,
    int8_bp=False,
):
    if isinstance(mod, relay.Function):
        mod = tvm.IRModule.from_expr(mod)
    assert isinstance(mod, tvm.IRModule)
    mod = relay.transform.InferType()(mod)
    pred = mod["main"].body

    ad = AutoDiff(debug=False, sparse_op_idx=sparse_op_idx)
    ad.int8_grad = int8_bp
    ad.compute_grad(mod["main"].body)

    names, gradients = ad.obtain_grads(filter_fn=filter_fn)
    expr = relay.Function(ad.vars, relay.Tuple(gradients))

    if keep_prediction:
        expr = relay.Function(
            ad.vars,
            relay.Tuple(
                [
                    pred,
                ]
                + gradients
            ),
        )
        names = [
            "fwd@output",
        ] + names

    bwd_mod = tvm.IRModule.from_expr(expr)

    # post processing
    bwd_mod = relay.transform.InferType()(bwd_mod)

    bwd_mod = tvm.transform.Sequential(
        [
            relay.transform.DeadCodeElimination(),
            relay.transform.ToGraphNormalForm(),
            relay.transform.FoldConstant(),
            relay.transform.SimplifyExpr(),
        ]
    )(bwd_mod)

    if return_gradient_tensors:
        return bwd_mod, names, gradients

    if not return_sparse_meta_info:
        return bwd_mod, names
    else:
        return bwd_mod, names, ad.sparse_update_meta_info


def torchscript_to_ir(scripted_model, shape_list):
    from compilation.utils import from_pytorch

    mod, params = from_pytorch(
        scripted_model,
        shape_list,
        use_parser_friendly_name=True,
    )
    names = [_.name_hint for _ in relay.analysis.all_vars(mod["main"].body)]
    return mod, params, names


if __name__ == "__main__":
    # ad = AutoDiff()
    # ad.compute_grad(mod['main'].body)
    pass
