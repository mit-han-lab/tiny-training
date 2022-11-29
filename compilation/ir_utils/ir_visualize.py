import numpy as np

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import tvm
from tvm import relay

from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor

import graphviz


def VisualizeIR(ir, path):
    import os

    v = MyVisistor()
    v.display_vars = True
    v.visualize(ir)
    os.makedirs(".tmp", exist_ok=True)
    v.dot.render(filename=path)
    print(f"Export to {path}")


class MyVisistor(ExprVisitor):
    def __init__(self, display_vars=False):
        super().__init__()
        self.dot = graphviz.Digraph("Mod Viz", comment="The Round Table")
        self.counter = 0
        self.names = set()
        self.id2name = dict()
        self.display_vars = display_vars

    def visualize(self, ast):
        if isinstance(ast, tvm.ir.module.IRModule):
            k = list(ast.functions.keys())[0]
            fname = str(k.name_hint)
            call = ast[fname].body
        elif isinstance(ast, relay.function.Function):
            call = ast.body
        elif isinstance(ast, tvm.relay.expr.Call):
            call = ast
        return self.visit(call)

    def visit_var(self, var):
        # print(var.name_hint)
        name = var.name_hint
        if self.display_vars:
            self.dot.node(
                name,
                # f'''<{var.name_hint}<BR/><FONT POINT-SIZE="8">{var.checked_type.shape} </FONT> >''',
                var.name_hint,
                fillcolor="#EEEEEE",
                style="filled",
                fontsize="8",
            )

    def visit_call(self, call):
        # recursively parse the AST
        self.visit(call.op)
        for a in call.args:
            self.visit(a)

        # Add function node
        name = str(call.op)
        # count = 0
        # while f"{name}-{count}" in self.names:
        #     count += 1
        count = self.counter
        self.counter += 1
        self.names.add(f"{name}-{count}")
        self.id2name[call.handle.value] = f"{name}-{count}"

        mfstr = f"{name}@{count}"
        fstr = ""
        attrs = call.attrs
        if attrs is not None:
            for k in attrs.keys():
                fstr += f"""{k}:{attrs[k]} <BR ALIGN="LEFT"/>"""
        self.dot.node(
            f"{name}-{count}",
            mfstr,
            shape="box3d",
            fontsize="16",
            style="filled",
            fillcolor="#335588",
            fontcolor="#FFFFFF",
            color="#000000",
        )

        weight_v = 5
        for idx, a in enumerate(call.args):
            w = 10
            if isinstance(a, relay.expr.Var):
                if self.display_vars:
                    self.dot.edge(a.name_hint, f"{name}-{count}")
            if isinstance(a, relay.expr.Call):
                # print(f"{name}-{count}", call.handle.value, self.id2name.keys())
                if idx != 0:
                    w = 2
                self.dot.edge(
                    self.id2name[a.handle.value],
                    f"{name}-{count}",
                    weight=str(w),
                    splines="polyline",
                )
            if isinstance(a, relay.expr.TupleGetItem):
                prev_node = self.id2name[a.tuple_value.handle.value]
                self.dot.edge(prev_node, f"{name}-{count}", weight=str(w))


if __name__ == "__main__":
    # net = nn.Sequential(
    #     nn.Conv2d(3, 3, 3, bias=None, padding=1),
    #     nn.BatchNorm2d(3),
    #     nn.Conv2d(3, 3, 3, bias=None, padding=1),
    #     nn.ReLU()
    # )
    # net = models.mobilenet_v2()
    # net = net.features

    # rs = 224
    # input_shape = [1, 3, rs, rs]
    # input_data = torch.randn(input_shape)
    # input_name = "input0"
    # shape_list = [(input_name, input_data.shape)]

    # scripted_model = torch.jit.trace(net, input_data).eval()
    # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    import sys

    ir_path = "ir_zoos/mcunet_quantize/fwd-1x3x128x128.ir"
    if len(sys.argv) >= 2:
        ir_path = sys.argv[-1]

    with open(ir_path, "r") as fp:
        code = fp.read()
    expr = tvm.parser.parse_expr(code)
    mod = tvm.IRModule.from_expr(expr)
    fname = ir_path.split("/")[-1].replace(".ir", "")
    v = MyVisistor()
    v.display_vars = True
    v.visualize(mod)
    import os

    os.makedirs(".tmp", exist_ok=True)
    v.dot.render(filename=f"./tmp/{fname}")
    print(f"Export to ./tmp/{fname}.pdf")
