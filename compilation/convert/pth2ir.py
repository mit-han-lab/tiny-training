from logging import warning
import torch

import tvm
from tvm import relay
from .pth_utils import nn_seq_to_ir, nn_module_to_ir
from ..autodiff.auto_diff import appending_loss, bias_only, compute_autodiff


def pth_model_to_ir(model, input_res=(1, 3, 80, 80), num_classes=0):
    out, tot_args, export_params, op_idx = nn_module_to_ir(model, input_res=input_res)

    real_params = {}
    scale_params = {}
    for k, v in export_params.items():
        if k[0].isdigit():
            k = "v" + k
        if k.endswith("x_scale") or k.endswith("y_scale"):
            scale_params[k] = float(v.numpy())
        else:
            real_params[k] = v

    expr = relay.Function(tot_args, out)
    from compilation.ir_utils import ChangeName

    new_expr = ChangeName().visit(expr)
    fwd_mod = tvm.IRModule.from_expr(new_expr)

    if num_classes <= 0:
        fwd_mod = relay.transform.InferType()(fwd_mod)
        return fwd_mod, real_params, scale_params, op_idx
    else:

        out = relay.reshape(out, newshape=[0, 0])
        out = relay.cast(out, dtype="float32")
        expr = relay.Function(tot_args, out)

        new_expr = ChangeName().visit(expr)
        mod = tvm.IRModule.from_expr(new_expr)
        mod, params, names = appending_loss(
            mod, real_params, "label", label_shape=[1, num_classes]
        )
        fwd_mod_with_loss = relay.transform.InferType()(mod)

        return fwd_mod_with_loss, real_params, scale_params, op_idx


def generated_backward_graph(mod, op_idx, method, sparse_bp_config=None, int8_bp=True):
    def full_bp(v, g):
        vname = v.name_hint
        if not ("_weight" in vname or "_bias" in vname):
            return False
        if "input" in vname:
            return False
        if "label" in vname:
            return False
        if "_weight" in vname:
            return True
        if "_bias" in vname:
            return True

        return False

    def last_only(v, g):
        vname = v.name_hint
        if "input" in vname:
            return False
        if "label" in vname:
            return False
        if not ("_weight" in vname or "_bias" in vname):
            return False
        idx = int(vname.split("_")[0].replace("v", ""))
        if idx in [
            op_idx,
        ]:
            return True
        return False

    def bias_only(v, g):
        vname = v.name_hint
        if "input" in vname:
            return False
        if "label" in vname:
            return False
        if not ("_weight" in vname or "_bias" in vname):
            return False
        idx = int(vname.split("_")[0].replace("v", ""))
        if idx in [
            op_idx,
        ]:
            return True
        if "_bias" in vname:
            return True

        return False

    assert method in ["bias_only", "last_only", "sparse_bp", "full_bp"]
    if method in ["bias_only", "last_only", "full_bp"]:
        if sparse_bp_config is not None:
            import warnings

            warnings.warn(
                "bias_only or last_only does not require `sparse_bp_config` arg."
            )

        if method == "bias_only":
            method_fn = bias_only
        elif method == "last_only":
            method_fn = last_only
        elif method == "full_bp":
            method_fn = full_bp
        else:
            raise NotImplementedError

        bwd_mod, bwd_names = compute_autodiff(mod, filter_fn=method_fn, int8_bp=int8_bp)
        return bwd_mod, bwd_names
    elif method == "sparse_bp":
        assert sparse_bp_config
        from ..ir_utils import ir_scan_op

        total_convs = ir_scan_op(mod["main"])["nn.mcuconv2d"]
        # build sparse bp config
        sparse_op_idx = {}
        for idx, r in zip(
            sparse_bp_config["manual_weight_idx"],
            sparse_bp_config["weight_update_ratio"],
        ):
            if r <= 0:
                continue
            sparse_op_idx[total_convs - idx] = r

        def get_sparse_bp_fn():
            tot_bias = sparse_bp_config["n_bias_update"]
            bias_count = 0
            tot_modules = total_convs

            def sparse_bp(var, grad_info):
                vname = var.name_hint
                grad, is_sparse = grad_info
                if "input" in vname:
                    return False
                if "label" in vname:
                    return False
                if not ("_weight" in vname or "_bias" in vname):
                    return False
                idx = int(vname.split("_")[0].replace("v", ""))
                # if (op_idx - idx) <= config["n_bias_update"] and "_bias" in vname:
                nonlocal tot_bias, bias_count
                if "_bias" in vname:
                    bias_count += 1
                    if (tot_modules - bias_count) <= tot_bias:
                        return True
                    else:
                        return False
                if is_sparse and "_weight" in vname:
                    return True
                if f"{op_idx}_" in vname:
                    return True
                return False

            return sparse_bp

        bwd_mod, bwd_names, sparse_meta_info = compute_autodiff(
            mod,
            filter_fn=get_sparse_bp_fn(),
            sparse_op_idx=sparse_op_idx,
            return_sparse_meta_info=True,
            int8_bp=int8_bp,
        )

        from collections import Counter

        update_counter = Counter()
        for name in bwd_names:
            if "_bias" in name:
                update_counter["bias"] += 1
            if "_weight" in name:
                update_counter["weight"] += 1
        print("total update ", update_counter)
        return bwd_mod, bwd_names, sparse_meta_info
    else:
        raise NotImplementedError
