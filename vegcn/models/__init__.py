from .gcn_v import gcn_v

__factory__ = {
    'gcn_v': gcn_v,
}


def build_model(name, *args, **kwargs):
    if name not in __factory__:
        raise KeyError("Unknown model:", name)
    return __factory__[name](*args, **kwargs)
