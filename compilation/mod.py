import pickle
from mmap import mmap
from multiprocessing.context import assert_spawning
import os, os.path as osp

import numpy as np

import tvm
from tvm import relay, te
from tvm.contrib import graph_executor

import warnings

def mod_save(
    mod,
    params=None,
    meta=None,
    path=".models/sample_net",
    mod_name="mod.ir",
    param_name="weights.params",
):
    os.makedirs(path, exist_ok=True)
    with open(osp.join(path, mod_name), "w") as fp:
        fp.write(str(mod))

    if params is not None:
        with open(osp.join(path, param_name), "wb") as fp:
            fp.write(tvm.runtime.save_param_dict(params))

    if meta is not None:
        with open(osp.join(path, mod_name.replace(".ir", ".pkl")), "wb") as fp:
            # fp.write(str(mod["main"]))
            pickle.dump(meta, fp)


def mod_load(
    path=".models/sample_net",
    mod_name="mod.ir",
    param_name="weights.params",
    meta=None,
    adapt_output_info=False,
):
    SEMVER = '#[version = "0.0.5"]\n'

    assert osp.exists(osp.join(path, mod_name)), "%s not found " % osp.join(
        path, mod_name
    )
    with open(osp.join(path, mod_name), "r") as fp:
        code = fp.read()
        if adapt_output_info:
            lines = code.split("\n")
            segs = lines[0].split("->")
            segs[-1] = "{"
            line1 = "".join(segs)
            lines[0] = line1
            code = "\n".join(lines)

    metatable = None
    if meta is not None:
        print("Loading meta table information")
        with open(osp.join(path, meta), "rb") as fp:
            metatable = pickle.load(fp)
            metatable = {
                "relay.Constant": [
                    relay.const(_, dtype=str(_.dtype)) for _ in metatable
                ]
            }

    mod_expr = tvm.parser.parse(
        SEMVER + code,
        "from_string",
        None,
        metatable,
    )
    # mod = tvm.IRModule.from_expr(mod_expr)
    mod = mod_expr
    mod = relay.transform.InferType()(mod)

    params = None
    if not osp.exists(osp.join(path, param_name)):
        warnings.warn("%s not exist! Load none params" % osp.join(path, param_name))
    else:
        with open(osp.join(path, param_name), "rb") as fp:
            bin_params = fp.read()
        params = dict(tvm.runtime.load_param_dict(bin_params))
    return mod, params


class MRun:
    def __init__(self, mod=None, mpath=None, weights=None, wpath=None, target="llvm"):
        assert not mod or not mpath
        assert mod or mpath
        self.dev = tvm.cpu()
        if mod:
            self.mod = mod
        elif mpath:
            with open(mpath, "r") as fp:
                code = fp.read()
            SEMVER = '#[version = "0.0.5"]\n'
            mod_expr = tvm.parser.parse_expr(SEMVER + code)
            mod = tvm.IRModule.from_expr(mod_expr)
            mod = relay.transform.InferType()(mod)
            self.mod = mod

        self.vs = relay.analysis.all_vars(mod["main"])
        self.lib = relay.build(mod, target=target)
        self.g = graph_executor.GraphModule(self.lib["default"](tvm.cpu()))

        if wpath:
            print(f"weights loaded from {wpath}")
            with open(wpath, "rb") as fp:
                bin_params = fp.read()
            params = dict(tvm.runtime.load_param_dict(bin_params))
            new_params = {}
            for k, v in params.items():
                if k[0].isdigit():
                    k = "v" + k
                new_params[k] = v
            self.bind_data(new_params)
            self.new_params = new_params
        elif weights:
            self.bind_data(weights)
            self.new_params = weights

    def randomly_init_weights(self, loc=0, scale=1):
        tp = {}
        for idx, v in enumerate(self.vs):
            shape = [int(_) for _ in v.type_annotation.shape]
            dtype = str(v.type_annotation.dtype)
            # print(v.type_annotation.shape, v.type_annotation.dtype)
            p = np.ones(shape).astype(str(dtype))
            p = np.random.normal(loc=loc, scale=scale, size=shape).astype(str(dtype))
            tp[str(v.name_hint)] = p
        self.bind_data(tp)
        return tp

    def bind_data(self, data):
        if isinstance(data, np.ndarray):
            self.g.set_input(self.data_names[0], data)
        elif isinstance(data, dict):
            for k, v in data.items():
                try:
                    self.g.set_input(k, v)
                except (tvm._ffi.base.TVMError, ValueError):
                    t = self.g.get_input(k)
                    print(
                        f"Failed to set_input for |{k}|, feed-in: {v.shape, v.dtype}, expected {t.shape, t.dtype}\n"
                    )
                    # raise
                    exit(0)

    def __call__(self, data):
        self.bind_data(data)
        self.g.run()
        r = []
        for idx in range(self.g.get_num_outputs()):
            _r = self.g.get_output(idx)
            r.append(_r)
        return r


class ComputeDAG:
    def __init__(
        self,
        path,
        mod_name="mod.ir",
        param_name="weights.params",
        target="llvm",
        dev=tvm.cpu(0),
    ):
        self.path = path

        self.target = target
        self.dev = dev
        self.mod2lib = dict()
        self.lib2mod = dict()
        self.total_args = []

        mod, params = mod_load(path, mod_name, param_name)
        self.mod = mod
        if params is None:
            params = {}
        self.mod_params = params
        # print(param_name, self.mod_params.keys())
        # exit(0)

    def compile(self, mod_override=None, optimize=False):
        # with tvm.transform.PassContext(opt_level = opt_level):
        if mod_override is None:
            mod_to_build = self.mod
        else:
            mod_to_build = mod_override
        if optimize:
            mod_to_build = tvm.transform.Sequential(
                [
                    relay.transform.DeadCodeElimination(),
                    relay.transform.ToGraphNormalForm(),
                    relay.transform.FoldConstant(),
                    relay.transform.SimplifyExpr(),
                ]
            )(mod_to_build)

        lib = relay.build(mod_to_build, target=self.target, params=self.mod_params)
        lib_params = lib.get_params()

        vs = relay.analysis.all_vars(self.mod["main"])
        # the first elem is the input
        # self.input_name = vs[0].name_hint
        vname = [v.name_hint for v in vs][1:]
        func_args = []
        data_args = []
        total_args = []
        for arg in relay.analysis.all_vars(self.mod["main"]):
            vname = arg.name_hint
            if vname.startswith("x"):
                # TODO: this is a dirty fix to "let" assignments in TVMIR
                # TODO: find the proper binding in TVM underlying calls.
                continue
            if vname in self.mod_params.keys() or vname[1:] in self.mod_params.keys():
                # TODO: dirty fix to variable likes v0.weight
                func_args.append(arg)
            else:
                data_args.append(arg)
            # print("==" * 40)
            # print(vname, self.mod_params.keys() , func_args, data_args, sep="\n")
            total_args.append(arg)

        self.total_args = total_args
        self.data_args = data_args
        self.func_args = func_args
        print(f"data_args: @{len(data_args)}", [_.name_hint for _ in data_args])
        print(f"func_args: @{len(func_args)}", [_.name_hint for _ in func_args])

        # check vars and matched shape
        assert len(func_args) <= len(
            lib_params.keys()
        ), f"{len(func_args)}|{len(lib_params.keys())}\n{func_args}\n{lib_params.keys()}"

        for idx, args in enumerate(func_args):
            v = args.name_hint
            p1 = self.mod_params[v]
            p2 = lib_params["p" + str(idx)]
            # print(p1.shape, p2.shape, p1.shape ==  p2.shape)
            assert (
                p1.shape == p2.shape
            ), f"Shape mismatch for |{v}|, expected: {p1.shape}, get {p2.shape}"
            self.mod2lib[v] = "p" + str(idx)
            self.lib2mod["p" + str(idx)] = v

        self.data_names = [_.name_hint for _ in data_args]
        self.lib = lib
        self.lib_params = lib.get_params()
        self.g = graph_executor.GraphModule(lib["default"](self.dev))

    def bind_data(self, data):
        if isinstance(data, np.ndarray):
            self.g.set_input(self.data_names[0], data)
        elif isinstance(data, dict):
            for k, v in data.items():
                assert k in self.data_names
                self.g.set_input(k, v)

    def __call__(self, data):
        self.bind_data(data)
        self.g.run()
        r = []
        for idx in range(self.g.get_num_outputs()):
            _r = self.g.get_output(idx)
            r.append(_r)
        return r

    def get_params(self):
        return self.mod_params

    def set_params(self, new_param: dict):
        new_lib_params = dict()
        for k, v in new_param.items():
            assert k in self.mod_params, f"[{k}] is unseen is previous parameters."
            new_k = self.mod2lib[k]
            new_v = tvm.nd.array(v, self.dev)
            new_lib_params[new_k] = new_v
            self.mod_params[k] = new_v
        self.g.load_params(tvm.runtime.save_param_dict(new_lib_params))

    def load(self, path):
        warnings.warn("DAG.load function is deprecated!")
        mod, params = mod_load(path)
        self.mod = mod
        self.mod_params = params

    def save(self, path):
        warnings.warn("DAG.save function is deprecated!")
        mod_save(self.mod, self.mod_params, self.path)
