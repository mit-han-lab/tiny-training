# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import json

import torch.nn as nn
from ..modules import *
from .base import MyNetwork
from ..modules.layers import ResidualBlock


__all__ = ['ProxylessNASNets']


def proxyless_base(net_config=None, n_classes=None, dropout_rate=None):
    assert net_config is not None, 'Please input a network config'
    net_config_json = json.load(open(net_config, 'r'))

    if n_classes is not None:
        net_config_json['classifier']['out_features'] = n_classes
    if dropout_rate is not None:
        net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)

    return net


class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if isinstance(self.feature_mix_layer, ConvLayer):
            x = self.feature_mix_layer(x)
            x = x.mean(3).mean(2)
        elif isinstance(self.feature_mix_layer, LinearLayer):
            x = x.mean(3).mean(2)
            x = self.feature_mix_layer(x)
        else:
            assert self.feature_mix_layer is None
            x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(ResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if isinstance(m.conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.conv.point_linear.bn.weight.data.zero_()

    def load_state_dict(self, state_dict, strict=True):
        # fix the naming inconsistency before and after code refactor
        state_dict = {k.replace('.mobile_inverted_conv.', '.conv.'): v for k, v in state_dict.items()}
        super().load_state_dict(state_dict, strict)
