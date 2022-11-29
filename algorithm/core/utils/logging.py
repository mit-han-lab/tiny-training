import os
import sys
import yaml
from .config import configs
from loguru import logger as _logger
from core.utils import dist


__all__ = ['logger']

_logger.remove()
_logger.add(sys.stdout,
            level='DEBUG',
            format=(
                # '<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> '
                '<level>{message}</level>')
            )


class ExpLogger:
    def init(self):
        assert configs.run_dir is not None, 'Empty run directory!'
        if dist.rank() == 0:
            # dumping running configs
            _path = os.path.join(configs.run_dir, 'config.yaml')
            with open(_path, 'w') as f:
                from .config import configs2dict
                yaml.dump(configs2dict(configs), f)

            # also dump running log to file
            _logger.add(os.path.join(configs.run_dir, 'exp.log'))

    @staticmethod
    def info(*args):
        _logger.info(*args)

    @staticmethod
    def debug(*args):
        _logger.debug(*args)


logger = ExpLogger()
