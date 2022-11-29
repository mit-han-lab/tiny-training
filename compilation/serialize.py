from attr import Attribute
import numpy as np

from collections import Counter

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relay
from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor
from pprint import pprint

from .mod import mod_load, mod_save, ComputeDAG
from .utils import from_pytorch, convert_ir_var


class SerializeVisitor(ExprVisitor):
    def __init__(self, params=dict(), meta=None, verbose=False):
        super().__init__()
        self.names = set()
        self.graph = []
        self.params = dict()
        self.old_params = params
        self.MAP = dict()
        self.CHILD_COUNT = Counter()
        self.verbose = verbose

        self.meta = meta
        self.output_count_idx = 0

    def visit_var(self, var: tvm.relay.expr.Var):
        # print(var.name_hint)
        name = str(var.name_hint)
        if "input" in name or "label" in name:
            return
        # if name[0].isdigit():
        #     name = "v" + name
        # print(name, var.checked_type.dtype)
        if self.old_params is not None and name in self.old_params:
            if self.verbose:
                print(f"[loaded] {str(var.name_hint)} successfully loaded")
            self.params[str(var.name_hint)] = self.old_params[name].numpy()
        else:
            print(
                f"[missing] {str(var.name_hint)} is not found in the params, filling ones {var.checked_type.shape}"
            )
            shape = convert_ir_var(var.checked_type.shape)
            dtype = convert_ir_var(var.checked_type.dtype)
            self.params[str(var.name_hint)] = np.ones(shape).astype(dtype)

    def visit_call(self, call):
        # global MAP
        # Recursively parse the AST
        self.visit(call.op)
        for a in call.args:
            if isinstance(a, relay.expr.Call):
                self.CHILD_COUNT[a.handle.value] += 1
            self.visit(a)

        # Perform recording
        op_info = dict()
        op_count = 0
        while f"op@{op_count}" in self.names:
            op_count += 1
        op_name = f"op@{op_count}"
        self.names.add(op_name)
        op_type = str(call.op)
        op_name = f"{op_type}@{op_count}"

        op_info = {
            "name": op_name,
            "type": op_type,
        }

        op_attrs = dict()
        if call.attrs is not None:
            try:
                for k in call.attrs.keys():
                    temp = call.attrs[k]
                    # print(type(temp))
                    op_attrs[k] = convert_ir_var(temp)
            except AttributeError:
                raise NotImplementedError(f"Attrs of {call.op} is not registered")
        op_info["attrs"] = op_attrs

        op_args = []
        arg_shapes = []
        # process inputs
        for a in call.args:
            meta = None
            if isinstance(a, relay.expr.Call):
                name, shape, dtype = self.MAP[a.handle.value]
                var_type = "activation"
            elif isinstance(a, relay.expr.Var):
                name = a.name_hint
                shape = a.type_annotation.shape
                dtype = a.type_annotation.dtype
                var_type = "parameter"

            elif isinstance(a, relay.expr.Constant):
                name = f"{op_name}-constant"
                shape = a.data.shape
                dtype = a.data.dtype
                var_type = "constant"
                meta = {}
                meta["data"] = a.data.numpy().tolist()
            else:
                print("==" * 40)
                print(call.op, a, type(a))
                print("==" * 40)
                raise NotImplementedError(f"{type(a)} is not supported yet.")

            shape = convert_ir_var(shape)
            dtype = convert_ir_var(dtype)

            # dtype = convert_ir_var(call.checked_type.dtype)
            if len(shape) == 0:
                # print("yes")
                shape = [
                    1,
                ]
            info = {
                "name": name,
                "var_type": var_type,
                "shape": shape,
                "dtype": dtype,
                "meta": meta,
            }
            arg_shapes.append(shape)
            op_args.append(info)

        op_info["inputs"] = op_args

        out_name = f"out_{op_name}"
        shape = convert_ir_var(call.checked_type.shape)
        dtype = convert_ir_var(call.checked_type.dtype)
        if len(call.checked_type.shape) == 0:
            shape = [
                1,
            ]

        self.MAP[call.handle.value] = out_name, shape, dtype
        # print(out_name, call.checked_type.shape, call.checked_type.dtype)
        name = out_name

        meta = {"children": self.CHILD_COUNT[call.handle.value]}
        if self.CHILD_COUNT[call.handle.value] == 0 and self.meta is not None:
            # is leaf node:
            # print("find lead node", self.output_count_idx)
            meta["output_info"] = self.meta["output_info"][self.output_count_idx]
            # meta["output_count_idx"] = self.output_count_idx
            self.output_count_idx += 1

        info = {
            "name": name,
            "var_type": "activation",
            "shape": shape,
            "dtype": dtype,
            "meta": meta,
        }
        op_info["outputs"] = [
            info,
        ]

        self.graph.append(op_info)
