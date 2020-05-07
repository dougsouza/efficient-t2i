import torch
import torch.nn.functional as F


def normal(device, normalize=False):
    def sampler_fn(*size):
        r = torch.randn(*size, dtype=torch.float32, device=device)
        if normalize:
            r /= torch.norm(r, dim=1, keepdim=True)
        return r
    return sampler_fn


def censored_normal(device, normalize=False):
    def sampler_fn(*size):
        r = torch.randn(*size, dtype=torch.float32, device=device)
        if normalize:
            r /= torch.norm(r, dim=1, keepdim=True)
        return F.relu_(r)
    return sampler_fn


def bernoulli(device, normalize=False):
    def sampler_fn(*size):
        r = torch.rand(*size, dtype=torch.float32, device=device)
        return torch.bernoulli(r)
    return sampler_fn


def uniform(device, normalize=False):
    def sampler_fn(*size):
        r = torch.zeros(*size, dtype=torch.float32, device=device).uniform_(-1, 1)
        if normalize:
            r /= torch.norm(r, dim=1, keepdim=True)
        return r
    return sampler_fn



__priordict__ = {
    'normal': normal,
    'censored_normal': censored_normal,
    'uniform': uniform,
    'bernoulli': bernoulli
}


def get_sampler_fn(prior, **kwargs):
    return __priordict__[prior](**kwargs)


def get_priors_names():
    return __priordict__.keys()
