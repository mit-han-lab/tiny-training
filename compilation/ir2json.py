import os, os.path as osp
import json

import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relay

from compilation.mod import mod_load, mod_save, ComputeDAG
from compilation.utils import from_pytorch
from compilation.serialize import SerializeVisitor

def translate_pth_net(
    net,
    name="SampleNet",
    input_shape=(1, 3, 32, 32),
    requires_grad=["2_weight", "2_bias"],
):
    input_data = torch.randn(input_shape)
    input_name = "input0"
    shape_list = [(input_name, input_data.shape)]

    scripted_model = torch.jit.trace(net, input_data).eval()
    mod, params = from_pytorch(
        scripted_model, shape_list, use_parser_friendly_name=True
    )
    mod = relay.transform.InferType()(mod)

    return translate_mod(mod, None, name, input_shape, requires_grad)


def test_translate_pytorch():
    net = nn.Sequential(
        nn.Conv2d(3, 3, 3, bias=None, padding=1), nn.Flatten(), nn.Linear(3072, 10)
    )
    translate_pth_net(net, "SampleNet", input_shape=(1, 3, 32, 32))


def translate_ir(
    path="example/SampleNet.ir",
    name=None,
    out_folder=".model/testproj"
):
    folder = osp.dirname(path)
    file = osp.basename(path)
    if osp.exists(path.replace(".ir", ".pkl")):
        print("Loading meta")
        mod, params = mod_load(folder, mod_name=file, meta=file.replace(".ir", ".pkl"))
    else:
        mod, params = mod_load(folder, mod_name=file)
    
    if name is None:
        name = file.split(".")[0]
    
    meta_path = osp.join(path.replace(".ir", ".meta"))
    meta_info = None
    if osp.exists(meta_path):
        meta_info = json.load(open(meta_path, "r"))
    else:
        warnings.warn(f"{meta_path} not found")
    mod = relay.transform.InferType()(mod)
    # print(meta_info)
    # print(type(params), params.keys())
    
    if params is None:
        new_params = None
    else:
        new_params = {}
        for k, v in params.items():
            n = k 
            if k[0].isdigit():
                n = "v" + k
            new_params[n] = params[k]
    return translate_mod(mod, new_params, name, meta=meta_info, out_folder=out_folder)


def translate_mod(
    mod,
    params=None,
    name="example",
    meta=None,
    out_folder=".model/testproj",
    dump_to_file = True
):
    import json, pickle
    from pprint import pprint

    ev = SerializeVisitor(params=params, meta=meta)
    ev.visit(mod["main"].body)

    if dump_to_file:
        os.makedirs(osp.dirname(osp.join(out_folder, f"{name}.ir")), exist_ok=True)
        with open(osp.join(out_folder, f"{name}-graph.json"), "w") as fp:
            json.dump(ev.graph, fp, indent=2)
        with open(osp.join(out_folder, f"{name}-params.pkl"), "wb") as fp:
            pickle.dump(ev.params, fp)
        with open(osp.join(out_folder, f"{name}.ir"), "w") as fp:
            fp.write(str(mod["main"]))
        print(f"Successfully export to {out_folder}")
    return ev.graph, ev.params, str(mod["main"])

if __name__ == "__main__":
    import os, sys

    mod_path = "tmp/lenet/bias_only-1x1x32x32.ir"
    # mod_path = "tmp/lenet/weights.param"

    if len(sys.argv) >= 2:
        mod_path = sys.argv[-1]

    assert osp.exists(mod_path), f"{mod_path} does not exists."
    param_path = osp.join(osp.dirname(mod_path), "weights.param")
    translate_ir(path=mod_path , out_folder=".model/testproj")

    # mod, params = mod_load("./", mod_name=mod_path)
    # meta = osp.join(mod_path.replace(".ir", ".meta"))
    # meta_info = None
    # if osp.exists(meta):
    #     meta_info = json.load(open(meta, "r"))
    # mod = relay.transform.InferType()(mod)
    # print(meta_info)
    # translate(mod, params=None, name="lenet", meta=meta_info)