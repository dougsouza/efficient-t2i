from .bigGAN_deep import Generator as biggan_deepG
from .bigGAN_deep import Discriminator as biggan_deepD
from .bigGAN import Generator as bigganG
from .bigGAN import Discriminator as bigganD





__gmodelsdict__ = {'biggan_deep': biggan_deepG,
                   'biggan': bigganG}

__dmodelsdict__ = {'biggan_deep': biggan_deepD,
                   'biggan': bigganD}



def get_dmodel(d_model, **kwargs):
    return __dmodelsdict__[d_model](**kwargs)

def get_gmodel(g_model, **kwargs):
    return __gmodelsdict__[g_model](**kwargs)

def get_dmodels_names():
    return __dmodelsdict__.keys()

def get_gmodels_names():
    return __gmodelsdict__.keys()
