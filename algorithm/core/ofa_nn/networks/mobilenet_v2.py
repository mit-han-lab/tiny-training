from .proxyless_nets import ProxylessNASNets
from ..modules import *
from ..utils import val2list, make_divisible

__all__ = ['MobileNetV2']


class MobileNetV2(ProxylessNASNets):

    def __init__(self, n_classes=1000, width_mult=1, dropout_rate=0.,
                 ks=None, expand_ratio=None, depth_param=None, stage_width_list=None,
                 # extended options
                 mix_layer='1x1', disable_keep_last_channel=False, last_channel=1280,
                 inverted_residual_setting=None, se_stages=None, fuse_blk1=False, fix_stem=False,
                 act_func='relu6', divisible_by=8, input_channel=32, img_channel=3):

        if ks is None:
            ks = 3
        if expand_ratio is None:
            expand_ratio = 6

        if not fix_stem:
            input_channel = make_divisible(input_channel * width_mult, divisible_by)
        if disable_keep_last_channel:
            last_channel = make_divisible(last_channel * width_mult, divisible_by)
        else:
            last_channel = make_divisible(last_channel * width_mult,
                                          divisible_by) if width_mult > 1.0 else last_channel

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [(1, 'nodw'), 16, 1, 1] if fuse_blk1 else [1, 16, 1, 1],
                [expand_ratio, 24, 2, 2],
                [expand_ratio, 32, 3, 2],
                [expand_ratio, 64, 4, 2],
                [expand_ratio, 96, 3, 1],
                [expand_ratio, 160, 3, 2],
                [expand_ratio, 320, 1, 1],
            ]

        if depth_param is not None:
            assert isinstance(depth_param, (int, list))
            if isinstance(depth_param, list):
                assert len(depth_param) == len(inverted_residual_setting)
                assert depth_param[0] == depth_param[-1] == 1
            for i in range(1, len(inverted_residual_setting) - 1):
                inverted_residual_setting[i][2] = depth_param[i] if isinstance(depth_param, list) else depth_param

        if stage_width_list is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][1] = stage_width_list[i]

        ks = val2list(ks, sum([n for _, _, n, _ in inverted_residual_setting]) - 1)
        assert len(ks) == sum([n for _, _, n, _ in inverted_residual_setting]) - 1
        ks = [3] + ks
        _pt = 0

        if se_stages is None:
            se_stages = [False] * len(inverted_residual_setting)
        else:
            assert len(se_stages) == len(inverted_residual_setting)

        act_func = val2list(act_func, len(inverted_residual_setting) + 2)

        # first conv layer
        first_conv = ConvLayer(
            img_channel, input_channel, kernel_size=3, stride=2, use_bn=True, act_func=act_func[0], ops_order='weight_bn_act'
        )
        # inverted residual blocks
        blocks = []
        for (t, c, n, s), use_se, act in zip(inverted_residual_setting, se_stages, act_func[1:-1]):
            if isinstance(t, tuple):
                assert t[1] == 'nodw'
                t = t[0]
                no_dw = True
            else:
                no_dw = False
            output_channel = make_divisible(c * width_mult, divisible_by)
            for i in range(n):
                stride = s if i == 0 else 1
                kernel_size = ks[_pt]
                _pt += 1
                conv = MBInvertedConvLayer(
                    in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride,
                    expand_ratio=t, use_se=use_se[i] if isinstance(use_se, list) else use_se, act_func=act,
                    no_dw=no_dw
                )
                if t > 1 and stride == 1:  # NOTICE: we enforce no residual for the first block
                    if input_channel == output_channel:
                        shortcut = IdentityLayer(input_channel, input_channel)
                    else:
                        shortcut = None
                else:
                    shortcut = None
                blocks.append(
                    ResidualBlock(conv, shortcut)
                )
                input_channel = output_channel
        # 1x1_conv before global average pooling
        self.mix_layer = mix_layer
        if mix_layer == '1x1':
            feature_mix_layer = ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func=act_func[-1])
            input_channel = last_channel
        elif mix_layer == 'group':
            feature_mix_layer = ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func=act_func[-1])
            input_channel = last_channel // 4
        elif mix_layer == 'lin':
            feature_mix_layer = ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='none')
            input_channel = last_channel
        elif mix_layer == 'fc':
            blocks[-1] = blocks[-1].conv.inverted_bottleneck
            input_channel = blocks[-1][0].out_channels
            feature_mix_layer = LinearLayer(input_channel, last_channel, bias=True, use_bn=False, act_func=act_func[-1])
            input_channel = last_channel
        elif mix_layer == 'trunc':
            blocks[-1] = blocks[-1].conv.inverted_bottleneck
            input_channel = blocks[-1][0].out_channels
            feature_mix_layer = None
        else:
            assert mix_layer is None or mix_layer.lower() == 'none'
            feature_mix_layer = None

        classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

        super(MobileNetV2, self).__init__(first_conv, blocks, feature_mix_layer, classifier)
