from .CUB_200_2011 import CUB200Dataset
from .Oxford102 import Oxford102Dataset



__loadersdict__ = {'CUB': CUB200Dataset,
                    'Oxford': Oxford102Dataset}


def get_loader(name, **kwargs):
    return __loadersdict__[name](**kwargs)


def get_loader_names():
    return __loadersdict__.keys()
