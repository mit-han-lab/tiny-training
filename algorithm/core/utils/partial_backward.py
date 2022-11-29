import copy
import torch
import numpy as np

activation_bits = 8
fc_bits = 0  # 32  # do not consider fc for now
weight_bits = 8
bias_bits = 32
momentum_bits = 0  # almost no momentum (per channel case)


# n_conv_backward:
# n_bias_update:
# n_weight_update:
# weight_update_ratio:
# weight_select_criteria:


def get_all_conv_ops(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    # sanity check, do not include the final fc layer
    return [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]

def get_all_conv_ops_with_names(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    # sanity check, do not include the final fc layer
    convs = []
    names = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            convs.append(m)
            names.append(name)
    return convs, names


def _is_depthwise_conv(conv):
    return conv.groups == conv.in_channels == conv.out_channels


def _is_pw1(conv):  # for mbnets
    return conv.out_channels > conv.in_channels and conv.kernel_size == (1, 1)  # could be groups


def parsed_backward_config(backward_config, model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    n_conv = len(get_all_conv_ops(model))
    # parse config (if None, update all)
    if backward_config['n_bias_update'] == 'all':
        backward_config['n_bias_update'] = n_conv
    else:
        assert isinstance(backward_config['n_bias_update'], int), backward_config['n_bias_update']

    # get the layer update index
    if backward_config['manual_weight_idx'] is not None:
        backward_config['manual_weight_idx'] = [int(p) for p in str(backward_config['manual_weight_idx']).split('-')]
    else:
        n_weight_update = backward_config.pop('n_weight_update')  # NOTE: we will not use this later
        if n_weight_update == 'all':  # the max layers to tune is the bias tune layers
            n_weight_update = backward_config['n_bias_update']
        else:
            assert isinstance(n_weight_update, int), n_weight_update
        backward_config['manual_weight_idx'] = sorted([n_conv - 1 - i_w for i_w in range(n_weight_update)])

    # for mobilenets only
    if backward_config['pw1_weight_only']:  # now filter out the non-pw1 weight
        all_convs = get_all_conv_ops(model)
        backward_config['manual_weight_idx'] = [idx for idx in backward_config['manual_weight_idx']
                                                if _is_pw1(all_convs[idx])]

    # sanity check: the weight update layers all update bias
    for idx in backward_config['manual_weight_idx']:
        assert idx in [n_conv - 1 - i_w for i_w in range(backward_config['n_bias_update'])]

    n_weight_update = len(backward_config['manual_weight_idx'])
    if backward_config['weight_update_ratio'] is None:
        backward_config['weight_update_ratio'] = [None] * n_weight_update
    elif isinstance(backward_config['weight_update_ratio'], (int, float)):  # single number
        assert backward_config['weight_update_ratio'] <= 1
        backward_config['weight_update_ratio'] = \
            [backward_config['weight_update_ratio']] * n_weight_update
    else:  # list
        backward_config['weight_update_ratio'] = [float(p) for p in backward_config['weight_update_ratio'].split('-')]
        assert len(backward_config['weight_update_ratio']) == n_weight_update
    # if we update weights, let's also update bias
    assert backward_config['n_bias_update'] >= n_weight_update
    return backward_config


def nelem_saved_for_backward(model, sample_input, backward_config, verbose=True, plot=False):
    """
    calculate the memory required when saving for backward
    :param model:
    :param sample_input:
    :param backward_config:
    :param verbose: whether to print log
    :param plot: whether to plot the distribution
    :return:
    """
    from quantize.quantized_ops_diff import ScaledLinear

    model = copy.deepcopy(model)
    model.eval()

    # firstly, let's get the input tensor size of all the operators
    def record_in_out_shape(m_, x, y):
        x = x[0]
        m_.input_shape = list(x.shape)
        m_.output_shape = list(y.shape)

    def add_activation_shape_hook(m_):
        m_.register_forward_hook(record_in_out_shape)

    def _zero_grad(m_):
        for p in m_.parameters():
            if p.grad is not None:
                p.grad = None

    model.apply(add_activation_shape_hook)

    with torch.no_grad():
        _ = model(sample_input)

    # now let's compute the saved memory size according to the backward config
    weight_size = []  # unit in bits
    momentum_size = []  # unit in bits
    activation_size = []  # unit in bits

    # firstly, let's count the usage of linear layer
    fc = model[-2]
    assert isinstance(fc, ScaledLinear), type(fc)
    weight_size.append((fc.weight.numel() + fc.bias.numel()) * fc_bits)
    momentum_size.append((fc.weight.numel() + fc.bias.numel()) * momentum_bits)
    if not len(fc.input_shape) == 2:
        assert len(fc.input_shape) == 4 and fc.input_shape[-1] == fc.input_shape[-2] == 1
    activation_size.append(fc.input_shape[1] * activation_bits)

    _zero_grad(model)

    conv_ops = get_all_conv_ops(model)[::-1]  # from back to front

    # now count the conv usage
    # register fake gradient for conv
    for conv in conv_ops:
        conv.weight.grad = torch.rand_like(conv.weight) * 100.
        conv.bias.grad = torch.rand_like(conv.bias) * 100.

    # apply backward config to remove gradient that we did not get
    apply_backward_config(model, backward_config)

    for conv in conv_ops:
        if conv.bias.grad is not None:  # this layer is updated
            # TODO: the mask and input might be counted twice; maybe we should fix this (or not, depends on impl.)?
            # if update, always update bias
            this_activation_size = np.product(conv.output_shape[1:]) * 1  # binary mask
            this_weight_size = conv.bias.numel() * bias_bits
            this_momentum_size = conv.bias.numel() * momentum_bits

            if conv.weight.grad is not None:

                if _is_depthwise_conv(conv):  # depthwise
                    weight_shape = conv.weight.shape  # o, 1, k, k
                    grad_norm = torch.norm(conv.weight.grad.data.view(weight_shape[0], -1), dim=1)
                    channels = (grad_norm > 0).sum().item()
                    this_activation_size += np.product(conv.input_shape[2:]) * channels * activation_bits
                    this_weight_size += (channels * weight_shape[2] * weight_shape[3]) * weight_bits
                    this_momentum_size += (channels * weight_shape[2] * weight_shape[3]) * momentum_bits
                else:
                    weight_shape = conv.weight.shape  # o, i, k, k
                    if conv.groups == 1:  # normal conv
                        grad_norm = torch.norm(conv.weight.grad.data.permute(1, 0, 2, 3).reshape(weight_shape[1], -1), dim=1)
                        channels = (grad_norm > 0).sum().item()
                        weight_elem = weight_shape[0] * channels * weight_shape[2] * weight_shape[3]
                    else:  # group conv (lite residual)
                        channels = conv.in_channels  # save all input channels
                        weight_elem = conv.weight.data.numel()  # update all weights

                    this_activation_size += np.product(conv.input_shape[2:]) * channels * activation_bits
                    this_weight_size += weight_elem * weight_bits
                    this_momentum_size += weight_elem * momentum_bits

            weight_size.insert(0, this_weight_size)
            momentum_size.insert(0, this_momentum_size)
            activation_size.insert(0, this_activation_size)

    del model

    total_weight_size = sum(weight_size)
    total_momentum_size = sum(momentum_size)
    total_activation_size = sum(activation_size)
    total_usage = total_weight_size + total_momentum_size + total_activation_size

    if verbose:
        print('weight', weight_size)
        print('momentum', momentum_size)
        print('activation', activation_size)
        print('memory usage in kB:')
        print('weight: {:.0f}kB, momentum: {:.0f}kB, activation: {:.0f}kB'.format(
            total_weight_size / 1024 / 8, total_momentum_size / 1024 / 8, total_activation_size / 1024 / 8
        ))
        print('total: {:.0f}kB'.format(total_usage / 1024 / 8))

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(weight_size)
        plt.plot(momentum_size)
        plt.plot(activation_size)
        plt.plot([a + b + c for a, b, c in zip(weight_size, momentum_size, activation_size)])
        plt.legend(['weight', 'momentum', 'activation', 'all'])
        plt.savefig('distribution.png')

    return total_usage


def prepare_model_for_backward_config(model, backward_config, verbose=True):
    def _get_conv_w_norm(_conv):
        _o, _i, _h, _w = _conv.weight.shape
        if _is_depthwise_conv(_conv):
            w_norm = torch.norm(_conv.weight.data.view(_o, -1), dim=1)
        else:
            w_norm = torch.norm(_conv.weight.data.permute(1, 0, 2, 3).reshape(_i, -1), dim=1)
        assert w_norm.numel() == _conv.in_channels
        return w_norm
    # assume sorted
    assert backward_config['manual_weight_idx'] == sorted(backward_config['manual_weight_idx']), backward_config['manual_weight_idx']

    # select the channels to be update
    if all([r is None for r in backward_config['weight_update_ratio']]):  # all channel update; return
        return
    else:
        total_keep_channel = 0
        conv_ops = get_all_conv_ops(model)
        ratio_ptr = 0
        for i_conv, conv in enumerate(conv_ops):  # from input to output
            if i_conv in backward_config['manual_weight_idx']:  # the weight is updated for this layer
                keep_ratio = backward_config['weight_update_ratio'][ratio_ptr]
                ratio_ptr += 1
                if keep_ratio <= 1:
                    n_keep = int(conv.in_channels * keep_ratio)
                else:
                    assert isinstance(keep_ratio, int)
                    n_keep = keep_ratio
                total_keep_channel += n_keep
                if backward_config['weight_select_criteria'] == 'magnitude+':
                    w_norm = _get_conv_w_norm(conv)
                    keep_idx = torch.argsort(-w_norm)[:n_keep]
                    keep_mask = torch.zeros_like(w_norm)
                    keep_mask[keep_idx] = 1.
                elif backward_config['weight_select_criteria'] == 'magnitude-':
                    w_norm = _get_conv_w_norm(conv)
                    keep_idx = torch.argsort(w_norm)[:n_keep]
                    keep_mask = torch.zeros_like(w_norm)
                    keep_mask[keep_idx] = 1.
                elif backward_config['weight_select_criteria'] == 'random':
                    w_norm = _get_conv_w_norm(conv)  # not used actually
                    keep_idx = torch.randperm(conv.in_channels)[:n_keep]
                    keep_mask = torch.zeros_like(w_norm)
                    keep_mask[keep_idx] = 1.
                else:
                    raise NotImplementedError
                conv.register_buffer('keep_mask', keep_mask)
        avg_channel = total_keep_channel / len(backward_config['weight_update_ratio'])
        if verbose:
            print(f' * Total update channels: {total_keep_channel}; average per layer: {avg_channel}')


def apply_backward_config(model, backward_config):
    # 1. fc is always updated; no need to change
    # 2. delete gradient for some conv_ops
    n_w_trained = 0
    conv_ops = get_all_conv_ops(model)[::-1]
    ratio_ptr = len(backward_config['manual_weight_idx']) - 1
    for i_conv, conv in enumerate(conv_ops):  # back to front
        if i_conv < backward_config['n_bias_update']:
            real_idx = len(conv_ops) - i_conv - 1
            train_this_conv = real_idx in backward_config['manual_weight_idx']

            if train_this_conv and backward_config['pw1_weight_only']:
                assert _is_pw1(conv), conv

            if train_this_conv:
                n_w_trained += 1
                if backward_config['weight_update_ratio'][ratio_ptr] is not None:
                    # apply sub channel gradient
                    if _is_depthwise_conv(conv):
                        conv.weight.grad.data = conv.weight.grad.data * conv.keep_mask.view(-1, 1, 1, 1)
                    else:
                        conv.weight.grad.data = conv.weight.grad.data * conv.keep_mask.view(1, -1, 1, 1)
                    ratio_ptr -= 1
            else:  # only update bias; no weight
                conv.weight.grad = None
        else:  # do not even update
            conv.weight.grad = None
            conv.bias.grad = None
    assert n_w_trained == len(backward_config['manual_weight_idx']), \
        (n_w_trained, len(backward_config['manual_weight_idx']))

    # if backward_config['freeze_fc']:
    #     from quantize.quantized_ops_diff import ScaledLinear
    #     assert isinstance(model.module[-2], ScaledLinear)
    #     for p in model.module[-2].parameters():
    #         p.grad = None


def _test_nelem_saved_for_backward():
    from core.utils.config import configs, load_config_from_file
    from core.model.model_entry import build_model
    load_config_from_file('configs/default.yaml')
    configs.net_config.net_name = 'mcu-mbv2-w0.35-nomix-in-150e'
    configs.net_config.mcu_head_type = 'fp'
    configs.data_provider.num_classes = 10  # just use 10 cases as an example
    model = build_model()
    sample_input = torch.randn(1, 3, 128, 128)
    backward_config = configs['backward_config']
    backward_config['n_bias_update'] = 'all'
    backward_config['n_weight_update'] = 'all'
    print(backward_config)
    # {'enable_backward_config': 0, 'n_bias_update': None, 'n_weight_update': None, 'weight_update_ratio': None,
    #  'weight_select_criteria': 'magnitude'}
    backward_config = parsed_backward_config(backward_config, model)
    prepare_model_for_backward_config(model, backward_config)
    nelem_saved_for_backward(model, sample_input, backward_config, verbose=True, plot=True)


def _get_nelem_curve():
    from core.utils.config import configs, load_config_from_file
    from core.model.model_entry import build_model
    import copy
    load_config_from_file('configs/default.yaml')
    configs.net_config.net_name = 'mcu-mcunet-5fps-in-300e'
    # configs.net_config.net_name = 'mcu-mbv2-w0.35-nomix-in-150e'
    # configs.net_config.net_name = 'mcu-proxyless-w0.3-nomix-in-300e'
    configs.net_config.mcu_head_type = 'fp'
    configs.data_provider.num_classes = 10  # just use 10 cases as an example
    model = build_model()
    print(model)
    exit()

    n_conv = len(get_all_conv_ops(model))
    print(n_conv)

    sample_input = torch.randn(1, 3, 128, 128)
    out = []
    # manual_weight_idx = [27]
    # {'enable_backward_config': 1, 'n_bias_update': 24, 'n_weight_update': 'all',
    #  'weight_update_ratio': [1.0, 1.0, 1.0, 0.25, 0.25
    #      , 0.25], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 1,
    #  'manual_weight_idx': [21, 24, 27, 33, 36, 39]}
    # for i in [n_conv - min(manual_weight_idx)]:
    for i in [39]:
        backward_config = copy.deepcopy(configs['backward_config'])
        backward_config['n_bias_update'] = 'all'
        backward_config['n_weight_update'] = 'all'
        # backward_config['pw1_weight_only'] = 1
        # backward_config['manual_weight_idx'] = '-'.join([str(s) for s in [18, 23, 27, 30, 39, 29]])
        # backward_config['weight_update_ratio'] = '-'.join([str(s) for s in [0.5, 0.25, 0.5, 1, 1, 0.125]])
        # backward_config['weight_update_ratio'] = 0.25
        backward_config = parsed_backward_config(backward_config, model)
        prepare_model_for_backward_config(model, backward_config)
        n_elem = nelem_saved_for_backward(model, sample_input, backward_config, verbose=True, plot=False)
        out.append(n_elem)
        print(i, round(n_elem / 8 / 1024))
    print(out)
    print('in kb:', [int(round(o / 8 / 1024, 0)) for o in out])


if __name__ == '__main__':
    _test_nelem_saved_for_backward()
    # _get_nelem_curve()
