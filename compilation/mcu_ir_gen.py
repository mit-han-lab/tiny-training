import os
import os.path as osp
import json
from compilation.convert import (
    build_quantized_mcunet,
    build_quantized_mbv2,
    build_quantized_proxyless,
    pth_model_to_ir,
    generated_backward_graph
)

# some configs
model_name = "mcunet"
rs = 128
num_classes = 10
int8_bp = False

# convert pytorch model to forward graph
if model_name == "mbv2":
    path = "ir_zoos/mbv2_quantize"
    model, _ = build_quantized_mbv2(num_classes=num_classes)
    sparse_update_config = {
        "49kb": {
            'enable_backward_config': 1, 'n_bias_update': 16, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 0, 0.25, 0.125, 0.125, 0.125, 0.125], 'manual_weight_idx': [36, 39, 40, 41, 42, 45, 48, 49], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "73kb": {
            'enable_backward_config': 1, 'n_bias_update': 20, 'n_weight_update': 0, 'weight_update_ratio': [0.125, 0.5, 0.5, 1, 0.25, 0.125, 0.125, 1], 'manual_weight_idx': [32, 33, 36, 39, 41, 42, 45, 48], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "99kb": {
            'enable_backward_config': 1, 'n_bias_update': 25, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 1, 0.125, 0.5, 1], 'manual_weight_idx': [27, 30, 33, 36, 39, 42, 45, 48], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0,
        },
        "123kb": {
            'enable_backward_config': 1, 'n_bias_update': 31, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 1, 0.5, 1, 1], 'manual_weight_idx': [27, 30, 33, 36, 39, 42, 45, 48], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0,
        },
        "138kb": {
            'enable_backward_config': 1, 'n_bias_update': 34, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 1, 1, 1, 1], 'manual_weight_idx': [27, 30, 33, 36, 39, 42, 45, 48], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0,
        }
    }
elif model_name == "mcunet":
    path = "ir_zoos/mcunet_quantize"
    model, _ = build_quantized_mcunet(num_classes=num_classes)
    sparse_update_config = {
        "49kb": {
            "enable_backward_config": 1, "n_bias_update": 20, "n_weight_update": 0, "weight_update_ratio": [0, 0.25, 0.5, 0.5, 0, 0], "manual_weight_idx": [23, 24, 27, 30, 33, 39], "weight_select_criteria": "magnitude+", "pw1_weight_only": 0,
        },
        "74kb": {
            'enable_backward_config': 1, 'n_bias_update': 21, 'n_weight_update': 0, 'weight_update_ratio': [1, 0, 1, 0, 0.5, 1], 'manual_weight_idx': [21, 23, 24, 26, 27, 30], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "99kb": {
            'enable_backward_config': 1, 'n_bias_update': 22, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 0.125, 0.25], 'manual_weight_idx': [21, 24, 27, 30, 36, 39], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "124kb": {
            'enable_backward_config': 1, 'n_bias_update': 24, 'n_weight_update': 0, 'weight_update_ratio': [0.25, 1, 1, 1, 0.5, 0.5], 'manual_weight_idx': [21, 24, 27, 30, 33, 39], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "148kb": {
            'enable_backward_config': 1, 'n_bias_update': 23, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 1, 0.5], 'manual_weight_idx': [21, 24, 27, 30, 36, 39], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        }
    }
elif model_name == "proxyless":
    path = "ir_zoos/proxyless_quantize"
    model, _ = build_quantized_proxyless(num_classes=num_classes)
    sparse_update_config = {
        "49kb": {
            'enable_backward_config': 1, 'n_bias_update': 21, 'n_weight_update': 0, 'weight_update_ratio': [0.25, 1, 0, 1, 0, 0.125, 0.25, 0.25], 'manual_weight_idx': [39, 42, 44, 45, 50, 51, 54, 57], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "74kb": {
            'enable_backward_config': 1, 'n_bias_update': 28, 'n_weight_update': 0, 'weight_update_ratio': [0.5, 0.25, 1, 1, 1, 0.5, 0.25, 0.5], 'manual_weight_idx': [33, 36, 39, 42, 45, 51, 54, 57], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "98kb": {
            'enable_backward_config': 1, 'n_bias_update': 25, 'n_weight_update': 0, 'weight_update_ratio': [1, 0.5, 1, 1, 0.25, 0.5, 1, 1], 'manual_weight_idx': [36, 39, 42, 45, 48, 51, 54, 57], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "120kb": {
            'enable_backward_config': 1, 'n_bias_update': 32, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 0.5, 1, 1, 1], 'manual_weight_idx': [36, 39, 42, 45, 48, 51, 54, 57], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "148kb": {
            'enable_backward_config': 1, 'n_bias_update': 45, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 1, 1, 1, 1], 'manual_weight_idx': [36, 39, 42, 45, 48, 51, 54, 57], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        }
    }
fwd_mod, real_params, scale_params, op_idx = pth_model_to_ir(model, input_res=[1, 3, rs, rs], num_classes=num_classes)

from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor
from tvm import relay

if int8_bp:
    from compilation.autodiff.int8_grad import *
    path += "_int8grad"

os.makedirs(path, exist_ok=True)
with open(f"{path}/scale.json", "w") as fp:
    json.dump(scale_params, fp, indent=2)

from compilation.mod import MRun, mod_save
from tvm.relay.expr_functor import ExprMutator
class ExtractMetaConstants(ExprMutator):
    # Dirty fix for unknown relay.Const[Meta] occurance.
    def __init__(self):
        super().__init__()
        self.constants = []

    def visit_constant(self, const: relay.expr.Constant):
        new_const = relay.const(const.data.numpy())
        np_data = const.data.numpy()
        print("--" * 40)
        print(const, type(const), const.data, np_data, np_data.shape, np_data.size)
        if np_data.size == 1:
            value = np_data.item()
            #TODO: inherit value dtype
            new_const = relay.const(value, dtype=str(np_data.dtype))
        print(new_const)
        # input()
        if "meta" in str(const):
            self.constants.append(np_data)
        return new_const

    def extract_constants(self, func):
        expr = self.visit(func)
        return expr, self.constants

def extract_const_from_mod(mod):
    func = mod['main']
    new_func, consts = ExtractMetaConstants().extract_constants(func)
    return consts

fshape_str = "x".join([str(_) for _ in [1, 3, rs, rs]])
mod_save(fwd_mod, params=real_params, path=path, mod_name=f"fwd-{fshape_str}.ir")

method = "last_only"
for method in ["last_only",  "full_bp"]:
    bwd_mod, bwd_names = generated_backward_graph(fwd_mod, op_idx, method=f"{method}", int8_bp=int8_bp)
    meta_info = {
        "output_info" : bwd_names,
    }

    consts =  extract_const_from_mod(bwd_mod)
    print(consts, len(consts))
    mod_save(
        bwd_mod,
        None,
        path=f"{path}",
        mod_name=f"{method}-{fshape_str}.ir",
        meta=consts
    )


    mod_save(
        bwd_mod,
        None,
        path=f"{path}",
        mod_name=f"{method}-{fshape_str}.ir",
    )
    with open(osp.join(path, f"{method}-{fshape_str}.meta"), "w") as fp:
        json.dump(
            meta_info,
            fp,
            indent=2,
        )

    print("Saving all information for ", f"{method}-{fshape_str}.ir",)

for mem, cfg in sparse_update_config.items():
    # derive the correspoding backward graph
    # method has to either `last_only`, `bias_only`, `sparse_bp`
    print(mem)
    bwd_mod, bwd_names, sparse_meta_info = generated_backward_graph(fwd_mod, op_idx, method="sparse_bp", sparse_bp_config=cfg, int8_bp=int8_bp)
    meta_info = {
        "output_info" : bwd_names,
        "sparse_update_info": sparse_meta_info,
    }
    # expr = bwd_mod['main']
    # from graph_tools.extract_const import extract_constants
    # expr, const_dict = extract_constants(expr)
    # bwd_mod = tvm.IRModule.from_expr(expr)
    consts =  extract_const_from_mod(bwd_mod)

    mod_save(
        bwd_mod,
        None,
        path=f"{path}",
        mod_name=f"sparse_bp-{mem}-{fshape_str}.ir",
        meta=consts
    )
    with open(osp.join(path, f"sparse_bp-{mem}-{fshape_str}.meta"), "w") as fp:
        json.dump(
            meta_info,
            fp,
            indent=2,
        )
    print(bwd_names)



