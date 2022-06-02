"""
Combination of utility functions from multiple sources

1-
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146
    The code reproduces *exactly* it's lua version:
    https://github.com/szagoruyko/wide-residual-networks
    2016 Sergey Zagoruyko

2-
    https://github.com/lwneal/counterfactual-open-set/blob/master/generativeopenset/vector.py
"""

import torch
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from nested_dict import nested_dict


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    assert isinstance(device_ids, list)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True

def clamp_to_unit_sphere(x, components=1):
    # If components=4, then we normalize each quarter of x independently
    # Useful for the latent spaces of fully-convolutional networks
    batch_size, latent_size = x.shape
    latent_subspaces = []
    for i in range(components):
        step = latent_size // components
        left, right = step * i, step * (i+1)
        subspace = x[:, left:right].clone()
        norm = torch.norm(subspace, p=2, dim=1)
        subspace = subspace / norm.expand(1, -1).t()  # + epsilon
        latent_subspaces.append(subspace)
    # Join the normalized pieces back together
    return torch.cat(latent_subspaces, dim=1)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x
