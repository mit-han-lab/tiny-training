from collections import Counter
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relay
from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor
from pprint import pprint

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

class ChangeName(ExprMutator):
    def visit_var(self, var):
        vname = var.name_hint
        shape = var.type_annotation.shape
        dtype = var.type_annotation.dtype
        if vname[0].isdigit():
            return relay.var("v" + vname, shape=shape, dtype=dtype)
        return var

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
        return Call(new_fn, args, call.attrs)
        
class OPCounter(ExprVisitor):
    def __init__(self, expr):
        super().__init__()
        self.counter = Counter()
        self.visit(expr)

    def visit_var(self, var: tvm.relay.expr.Var):
        return super().visit_var(var)

    def visit_call(self, call):
        # Recursively parse the AST
        self.visit(call.op)
        self.counter[str(call.op)] += 1
        for a in call.args:
            self.visit(a)


def ir_scan_op(expr):
    op = OPCounter(expr)
    return op.counter


def test_ir_scan_op():
    x = relay.var("x", shape=[1, 10])
    y = relay.var("y", shape=[1, 10])
    z = relay.var("z", shape=[1, 10])
    out = relay.add(x, y)
    out = relay.subtract(out, z)
    out = relay.multiply(out, z)
    expr = relay.Function([x, y, z], out)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)

    ir_scan_op(expr)


from tvm.relay.expr_functor import ExprMutator, Call


class RenameVar(ExprMutator):
    def __init__(self, fn: None):
        super().__init__()
        self.fn = fn

    def visit_var(self, var):
        vname = var.name_hint
        shape = var.type_annotation.shape
        dtype = var.type_annotation.dtype
        if self.fn:
            r = self.fn(vname)
            if r is not None:
                return relay.var(r, shape=shape, dtype=dtype)
        return var

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
        return Call(new_fn, args, call.attrs)


def rename_var(expr, fn=None):
    expr = RenameVar(fn).visit(expr)
    return expr


def test_rename_var():
    x = relay.var("x", shape=[1, 10])
    y = relay.var("y", shape=[1, 10])
    z = relay.var("z", shape=[1, 10])
    out = relay.add(x, y)
    out = relay.subtract(out, z)
    out = relay.multiply(out, z)
    expr = relay.Function([x, y, z], out)

    def fn(vname):
        # if vname[0].isdigit():
        if "x" in vname:
            return "test_var_123"
        return None

    new_expr = RenameVar(fn).visit(expr)
    print(new_expr)


class ReplaceOP(ExprMutator):
    def __init__(self, replace_fn=None):
        super().__init__()
        self.replace_fn = replace_fn

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))

        if self.replace_fn:
            r = self.replace_fn(call, args, call.attrs)
            if r:
                return r
        return Call(new_fn, args, call.attrs)


def replace_op(expr, fn):
    return ReplaceOP(replace_fn=fn).visit(expr)


def test_replace_op():
    x = relay.var("x", shape=[1, 10])
    y = relay.var("y", shape=[1, 10])
    z = relay.var("z", shape=[1, 10])
    out = relay.add(x, y)
    out = relay.subtract(out, z)
    out = relay.sum(out, axis=0)
    expr = relay.Function([x, y, z], out)

    def replace_fn(call, args, attrs):
        if str(call.op) == "sum":
            print("replacing sum to prod")
            return relay.prod(args[0], axis=attrs.axis)
        return None

    print(ReplaceOP(replace_fn=replace_fn).visit(expr))


class SimplifyReshape(ExprMutator):
    def __init__(self):
        super().__init__()
        self.op_idx = 0

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
        if str(call.op) == "reshape":
            # if the output shape totally matches, then skip
            # newshape = call.attrs.newshape
            print(
                f"{call.op} {self.op_idx} {call.attrs.newshape} args: {[type(_) for _ in call.args]}"
            )
            # current_shape = args[0].checked_type.shape
            current_shape = check_call_shape(args[0])
            output_shape = call.checked_type.shape
            if len(output_shape) == len(current_shape) and all(
                [a == b for a, b in zip(output_shape, current_shape)]
            ):
                return args[0]
        self.op_idx += 1
        return Call(new_fn, args, call.attrs)


def test_simplify_reshape():
    x = relay.var("x", shape=[1, 10])
    out = relay.reshape(x, newshape=[1, 10])
    expr = relay.Function([x], out)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    expr = mod["main"]
    print(expr)
    new_expr = SimplifyReshape().visit(expr)
    print(new_expr)


if __name__ == "__main__":
    test_simplify_reshape()
