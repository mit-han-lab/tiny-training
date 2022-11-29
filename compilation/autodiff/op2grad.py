GRAD_OP_MAP = {}


def register_gradient(op_name, level=10):
    def _register_fn(fn):
        global GRAD_OP_MAP
        GRAD_OP_MAP[op_name] = fn

        def _call(*args, **kwargs):
            return fn(*args, **kwargs)

        return _call

    return _register_fn
