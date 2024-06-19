# -*- coding: utf-8 -*-

import os
from tkinter import W
from turtle import forward
from numpy import float64
import torch
import torch.nn.functional as F


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clone_module(module, memo=None):
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


def detach_module(module):
    if not isinstance(module, torch.nn.Module):
        return
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            detached = module._parameters[param_key].detach_()
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()
    for module_key in module._modules:
        detach_module(module._modules[module_key])


def detach_new_module(module):
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                cloned = param.detach()
                clone._parameters[param_key] = cloned
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]

                cloned = buff.detach()
                clone._buffers[buffer_key] = cloned
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = detach_new_module(
                module._modules[module_key],
            )
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


class Safe_Cosine_Similarity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1, v2, dim=-1, keepdim=False, eps_forward: float = 1e-12, eps_backward: float = 1e-12):
        v1_norm = v1.norm(dim=dim, keepdim=True)
        v2_norm = v2.norm(dim=dim, keepdim=True)
        v1_normalized = torch.div(v1, v1_norm.clamp_min(eps_forward))
        v2_normalized = torch.div(v2, v2_norm.clamp_min(eps_forward))
        cos_sim = torch.sum(v1_normalized * v2_normalized, dim=dim, keepdim=keepdim)
        ctx.save_for_backward(v1_normalized, v2_normalized, cos_sim, v1_norm, v2_norm)
        ctx.keepdim, ctx.dim, ctx.eps_backward = keepdim, dim, eps_backward
        return cos_sim, v1_normalized, v2_normalized, v1_norm, v2_norm

    @staticmethod
    def backward(ctx, grad_cos_sim, grad_v1_normalized, grad_v2_normalized, grad_v1_norm, grad_v2_norm):
        v1_normalized, v2_normalized, cos_sim, v1_norm, v2_norm = ctx.saved_tensors
        eps_backward, keepdim, dim = ctx.eps_backward, ctx.keepdim, ctx.dim
        if not keepdim:
            cos_sim = cos_sim.unsqueeze(dim)
            grad_cos_sim = grad_cos_sim.unsqueeze(dim)
        grad_v1 = torch.div(v2_normalized * grad_cos_sim + grad_v1_normalized - (
                cos_sim * grad_cos_sim + torch.sum(grad_v1_normalized * v1_normalized, dim=-1,
                                                   keepdim=True)) * v1_normalized,
                            v1_norm.clamp_min(eps_backward))
        grad_v2 = torch.div(v1_normalized * grad_cos_sim + grad_v2_normalized - (
                cos_sim * grad_cos_sim + torch.sum(grad_v2_normalized * v2_normalized, dim=-1,
                                                   keepdim=True)) * v2_normalized,
                            v2_norm.clamp_min(eps_backward))
        grad_v1 += v1_normalized * grad_v1_norm
        grad_v2 += v2_normalized * grad_v2_norm
        return grad_v1, grad_v2, None, None, None, None


def safe_cosine_similarity(v1, v2, dim=-1, keepdim=False, eps_forward: float = 1e-7, eps_backward: float = 1e-7):
    return Safe_Cosine_Similarity.apply(v1, v2, dim, keepdim, eps_forward, eps_backward)[0]


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
