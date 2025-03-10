import os

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

module_dir = os.path.dirname(os.path.abspath(__file__))

from torch.utils.cpp_extension import load  # noqa

gn_op = load(
    name="gn_op",
    sources=[
        os.path.join(module_dir, "csrc/custom_gn.cpp"),
        os.path.join(module_dir, "csrc/gn_kernel.cu"),
    ],
    extra_cuda_cflags=[
        "-use_fast_math",
        "-extra-device-vectorization",
        "-extended-lambda",  # for gpu_kernel (although this isn't used in custom GN kernels)
        "-lineinfo",  # useful for profiling
        "-src-in-ptx",
    ],
    extra_cflags=[
        "-Ofast",  # needed or else GN NCHW from source is slower than nn.GroupNorm
        "-funroll-all-loops",
        "-march=native",
    ],
    is_python_module=False,
    verbose=True,
)


class GN_NHWC_Func(torch.autograd.Function):  # noqa
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, g: int, eps: float, activation: str):
        ctx.x_shape = x.shape

        x_flat = x.view(x.shape[0], x.shape[1], -1)
        x_out, means, rstds = torch.ops.gnop.fwd(x_flat, weight, bias, g, eps, activation)
        ctx.save_for_backward(x_flat, weight, bias, means, rstds)
        ctx.G = g
        ctx.activation = activation
        return x_out.view(ctx.x_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_flat, weight, bias, means, rstds = ctx.saved_tensors
        dy = dy.contiguous(memory_format=torch.channels_last).view(x_flat.shape)
        assert dy.stride() == x_flat.stride()
        dx, dgamma, dbeta = torch.ops.gnop.bwd(dy, x_flat, weight, bias, means, rstds, ctx.G, ctx.activation)
        return dx.view(ctx.x_shape), dgamma, dbeta, None, None, None


class GN_NHWC(nn.GroupNorm):  # noqa
    def __init__(self, num_groups: int, num_channels: int, activation="identity", **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)
        activation_to_code = {
            "identity": 0,
            "relu": 1,
            "swish": 2,
            "silu": 2,
            "gelu": 3,
            "gelu_tanh": 4,
        }
        self.activation = activation
        self.act_code = activation_to_code[activation]

    @torch._dynamo.disable
    def forward(self, x):
        # N, C, H, W = x.shape
        # x = x.view(x.shape[0], x.shape[1], -1)
        _ = self.num_groups
        if x.stride(1) == 1:  # channels last format
            # make sure the other dims in x are contiguous (e.g. shape (2, 3, 5, 9)
            # should have stride (135, 1, 27, 3) and not (135, 1, 3, 15))
            inner_dims = range(2, x.ndim)
            x_contiguous = x.permute(0, *inner_dims, 1).contiguous()
            inner_dims = range(1, x.ndim - 1)
            x = x_contiguous.permute(0, -1, *inner_dims)
            fwd_fn = GN_NHWC_Func.apply
        else:  # channels first, fall back to torch's GN
            x = x.contiguous()
            activations = [lambda x: x, F.relu, F.silu, F.gelu, lambda x: F.gelu(x, approximate="tanh")]
            act_fn = activations[self.act_code]
            fwd_fn = lambda x, w, b, g, eps, _act: act_fn(F.group_norm(x, g, w, b, eps))  # noqa

        if self.affine:
            return fwd_fn(x, self.weight, self.bias, self.num_groups, self.eps, self.act_code)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return fwd_fn(x, w, b, self.num_groups, self.eps, self.act_code)

    def extra_repr(self):
        return "{num_groups}, {num_channels}, eps={eps}, affine={affine}, activation={activation}".format(
            **self.__dict__
        )
