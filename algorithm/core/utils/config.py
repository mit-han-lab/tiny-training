import os
import yaml
import argparse
from easydict import EasyDict
from typing import Union

configs = EasyDict()


def load_config_from_file(file_path: str) -> None:
    def _iterative_update(dict1, dict2):
        for k in dict2:
            if k not in dict1:
                dict1[k] = dict2[k]
            else:  # k both in dict1 and dict2
                if isinstance(dict2[k], (dict, EasyDict)):
                    assert isinstance(dict1[k], (dict, EasyDict))
                    _iterative_update(dict1[k], dict2[k])
                else:
                    dict1[k] = dict2[k]

    global configs
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    # assert file_path.startswith('configs/') and file_path.endswith('.yaml'), \
    #     f'Invalid config path {file_path}'
    assert 'configs/' in file_path
    prefix = file_path.split('configs/')[0]
    file_path = file_path[len(prefix):]

    levels = file_path.split('/')
    for i_level in range(len(levels)):
        path = prefix + '/'.join(levels[:i_level + 1])
        if i_level == len(levels) - 1:
            cur_config_path = path
        else:
            cur_config_path = os.path.join(path, 'default.yaml')
        if os.path.exists(cur_config_path):
            with open(cur_config_path, 'r') as f:
                _config = yaml.safe_load(f)
                _iterative_update(configs, _config)


def update_config_from_args(args: Union[dict, argparse.Namespace]):
    global configs

    def _iterative_update(edict, new_k, new_v):
        for _k, _v in edict.items():
            if _k == new_k:
                edict[new_k] = new_v
                return True
        for _, _v in edict.items():
            if isinstance(_v, (dict, EasyDict)):
                _ret = _iterative_update(_v, new_k, new_v)
                if _ret:
                    return True
        return False

    if isinstance(args, argparse.Namespace):
        args = args.__dict__
    for k, v in args.items():
        if v is None or k == 'config':
            continue
        ret = _iterative_update(configs, k, v)
        if not ret:
            raise ValueError(f'ERROR: Updating args failed: cannot find key: {k}')


def parse_unknown_args(unknown):
    def _convert_value(_v):
        try:  # int
            return int(_v)
        except ValueError:
            pass
        try:  # float
            return float(_v)
        except ValueError:
            pass
        return _v  # string

    assert len(unknown) % 2 == 0
    parsed = dict()
    for idx in range(len(unknown) // 2):
        k, v = unknown[idx * 2], unknown[idx * 2 + 1]
        assert k.startswith('--')
        k = k[2:]
        v = _convert_value(v)
        parsed[k] = v
    return parsed


def update_config_from_unknown_args(unknown):
    parsed = parse_unknown_args(unknown)
    print(' * Getting extra args', parsed)
    update_config_from_args(parsed)


# iteratively convert easy dict to dictionary
def configs2dict(cfg):
    from easydict import EasyDict
    if isinstance(cfg, EasyDict):
        cfg = dict(cfg)
        key2cast = [k for k in cfg if isinstance(cfg[k], EasyDict)]
        for k in key2cast:
            cfg[k] = configs2dict(cfg[k])
        return cfg
    else:
        return cfg


if __name__ == '__main__':
    load_config_from_file('/home/jilin/workspace/clip_distill/configs/flowers102/resnet50.yaml')
    print(configs.run_config.n_epochs)
    update_config_from_args({'n_epochs': 100, 'dummy': 100})
    print(configs.run_config.n_epochs)
