import torch
from ...core.utils.config import configs 
from ...quantize.custom_quantized_format import build_quantized_network_from_cfg
from ...quantize.quantize_helper import create_scaled_head, create_quantized_head

__all__ = ['build_mcu_model']


def build_mcu_model():
    cfg_path = f"assets/mcu_models/{configs.net_config.net_name}.pkl"
    cfg = torch.load(cfg_path)
    
    model = build_quantized_network_from_cfg(cfg, n_bit=8)

    if configs.net_config.mcu_head_type == 'quantized':
        model = create_quantized_head(model)
    elif configs.net_config.mcu_head_type == 'fp':
        model = create_scaled_head(model, norm_feat=False)
    else:
        raise NotImplementedError

    return model
