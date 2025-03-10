# Exhaustive search to check if GN outputs are equal to reference.
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from tqdm import tqdm

from gnnhwc import GN_NHWC


class GN_Naive(nn.Module):  # noqa
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__()
        self.G = num_groups
        self.C = nc
        self.eps = 1e-05
        self.weight = nn.Parameter(torch.ones((nc,)))
        self.bias = nn.Parameter(torch.zeros((nc,)))
        self.x = None
        self.xnorm = None
        self.means = self.rstds = None

    def forward(self, x):
        n, c, h, w = x.shape
        self.x = x
        xr = x.reshape(n, self.G, h * w * c // self.G)
        means = xr.mean(dim=2, keepdim=True)
        rstds = torch.rsqrt(xr.var(dim=2, correction=0, keepdim=True) + self.eps)
        self.means = means[:, :, 0]
        self.rstds = rstds[:, :, 0]
        xnorm = (xr - means) * rstds
        self.xnorm = xnorm
        xnormr = xnorm.reshape(n, c, h, w) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return xnormr

    def bwd(self, dy):  # bwd pass for debugging
        n, c, h, w = dy.shape
        g = self.G
        d = c // g
        dyr = dy.view(n, g, d, h, w)
        xr = x.view(n, g, d, h, w)  # noqa
        dy_sum = dyr.sum((3, 4))
        xdy_sum = (dyr * xr).sum((3, 4))
        dy_gamma = (self.weight.view(1, g, d) * dy_sum).sum(2)
        xdy_gamma = (self.weight.view(1, g, d) * xdy_sum).sum(2)
        dweight = ((xdy_sum - self.means[:, :, None] * dy_sum) * self.rstds[:, :, None]).sum(0)  # noqa
        dbias = dy_sum.sum(0)  # noqa
        c1 = (self.means * dy_gamma - xdy_gamma) / (h * w * d) * self.rstds**3
        c2 = -self.means * c1 - dy_gamma * self.rstds / (h * w * d)
        dx = (  # noqa
            self.weight.view(1, g, d, 1, 1) * self.rstds.view(n, g, 1, 1, 1) * dyr
            + c1.view(n, g, 1, 1, 1) * xr
            + c2.view(n, g, 1, 1, 1)
        )


def red(text):
    return "\033[91m" + str(text) + "\033[0m"


def green(text):
    return "\033[92m" + str(text) + "\033[0m"


def yellow(text):
    return "\033[93m" + str(text) + "\033[0m"


def blue(text):
    return "\033[94m" + str(text) + "\033[0m"


def get_act_fn(act_str):
    return {
        "identity": lambda x: x,
        "silu": F.silu,
        "relu": F.relu,
        "gelu": F.gelu,
        "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    }[act_str]


def config_filter(x):  # returns true if config is valid
    _, dtype, b, c, h, w, g = x
    if c % g != 0:
        return False
    if (
        h * w == 1
    ):  # this causes an autograd problem where it gets confused since the tensor is both contiguous in NCHW/NHWC format
        return False

    dtype_size = torch.finfo(dtype).bits / 8
    estimated_mem_usage_gib = (25 * dtype_size * b * c * h * w) / 2**30  #  this is just a rough estimate, likely wrong
    if estimated_mem_usage_gib > 3:  # vram filter
        return False
    return True


bigx = torch.randn(128 * 1024 * 1024)


def check_params(params, verbose=True):
    vprint = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None  # noqa
    act_fn, dtype, b, c, h, w, g = params
    vprint(
        blue(
            f"output testing | ACT_FN: {act_fn} | DTYPE: {dtype} |B: {b:<2} | C: {c:<4} \
            | H: {h:<4} | W: {w:<4} | G: {g:<3}"
        )
    )
    xc = bigx[: b * c * h * w].reshape((b, c, h, w)).to(dtype).cuda()
    x = xc.to(memory_format=torch.channels_last)
    xc.requires_grad_(True)
    x.requires_grad_(True)
    torch.random.manual_seed(0)

    gn_test = GN_NHWC(g, c, activation=act_fn).cuda().to(dtype)
    # gn_test = nn.GroupNorm(G, C).cuda().to(DTYPE)

    gn_ref = nn.GroupNorm(g, c).cuda().to(dtype)
    act_fn = get_act_fn(act_fn)
    gn_naive = GN_Naive(g, c).cuda().to(dtype)
    with torch.no_grad():  # copy weights
        w = torch.randn((c,), dtype=dtype)
        b = torch.randn((c,), dtype=dtype)
        gn_ref.weight.copy_(w.detach())
        gn_ref.bias.copy_(b.detach())
        gn_test.weight.copy_(w.detach())
        gn_test.bias.copy_(b.detach())
        gn_naive.weight.copy_(w.detach())
        gn_naive.bias.copy_(b.detach())

    g_ref = act_fn(gn_ref(xc))
    g_test = gn_test(x)
    rand_dy = bigx[-b * c * h * w :].reshape((b, c, h, w)).to(dtype).cuda()
    rand_dy /= (
        rand_dy.numel() ** 0.5
    )  # to prevent false positive errors from ocurring because of really large magnitude losses

    err_params = False
    g_naive = g_naive_dx = None

    def vprint_err(x_ref, x_test, x_naive_fn, bwd=True, left_pad=0):
        nonlocal g_naive, g_naive_dx
        lpad = " " * left_pad
        with torch.no_grad():
            err = (x_ref - x_test).abs().max()

        if err < 10 * torch.finfo(dtype).resolution:
            vprint(green(f"{lpad}Negligible difference (err: {err:.2e}) found"))
            return False

        # second stage check: check both torch's GN and custom GN against a naive implementation to see if the error
        # could be caused by rounding errors
        if g_naive is None:
            g_naive = act_fn(gn_naive(xc))
        if bwd and g_naive_dx is None:
            xc.grad = None
            g_naive.backward(rand_dy)
            g_naive_dx = xc.grad

        with torch.no_grad():
            # we use a function because the variable that may be referred to may not be initialized until the function
            # is called, acts similar to a reference/pointer in java/c
            x_naive = x_naive_fn()
            err_ref_naive = (x_ref - x_naive).abs().max()
            err_test_naive = (x_test - x_naive).abs().max()

        if err_test_naive < 2 * err_ref_naive:
            vprint(
                yellow(
                    f"{lpad}Negligible difference (err: {err:.2e}, test-naive: {err_test_naive:.2e}, \
                    ref-naive: {err_ref_naive:.2e}) found"
                )
            )
            return False
        else:
            vprint(red(f"{lpad}Error: {err:.2e}, test-naive: {err_test_naive:.2e}, ref-naive: {err_ref_naive:.2e}"))
            return True

    vprint("  FORWARD")
    err_params = vprint_err(g_ref, g_test, lambda: g_ref, bwd=False, left_pad=4) or err_params
    vprint("  BACKWARD")
    xc.grad = None
    g_ref.backward(rand_dy)
    g_ref_dx = xc.grad

    x.grad = None
    g_test.backward(rand_dy)
    g_test_dx = x.grad

    vprint("    wrt X")
    err_params = vprint_err(g_ref_dx, g_test_dx, lambda: g_naive_dx, left_pad=6) or err_params
    vprint("    wrt weight")
    err_params = (
        vprint_err(gn_ref.weight.grad, gn_test.weight.grad, lambda: gn_naive.weight.grad, left_pad=6) or err_params
    )
    vprint("    wrt bias")
    err_params = vprint_err(gn_ref.bias.grad, gn_test.bias.grad, lambda: gn_naive.bias.grad, left_pad=6) or err_params

    return err_params


CUSTOM_INPUTS = [
    ("identity", torch.double, 1, 1, 2, 2, 1),
    ("identity", torch.double, 2, 3909, 5, 5, 3),
    ("identity", torch.double, 1, 2062, 5, 5, 1031),
    ("identity", torch.double, 3, 4096, 7, 7, 4),
    ("identity", torch.double, 1, 4096, 7, 7, 4),
    ("identity", torch.double, 2, 160, 8, 8, 160),
    ("identity", torch.double, 1, 3, 7, 7, 1),
    ("identity", torch.double, 1, 1, 4, 4, 1),
    ("identity", torch.double, 1, 128, 8, 8, 8),
    ("identity", torch.double, 2, 1280, 8, 8, 32),
    ("identity", torch.double, 2, 640, 16, 16, 32),
    ("identity", torch.double, 2, 2560, 8, 8, 32),
    ("identity", torch.double, 2, 1280, 16, 16, 32),
    ("identity", torch.double, 2, 320, 32, 32, 32),
    ("identity", torch.double, 2, 1920, 16, 16, 32),
    ("identity", torch.double, 2, 2560, 16, 16, 32),
    ("identity", torch.double, 2, 640, 32, 32, 32),
    ("identity", torch.double, 2, 960, 32, 32, 32),
    ("identity", torch.double, 2, 1280, 32, 32, 32),
    ("identity", torch.double, 2, 320, 64, 64, 32),
    ("identity", torch.double, 2, 1920, 32, 32, 32),
    ("identity", torch.double, 2, 640, 64, 64, 32),
    ("identity", torch.double, 8, 128, 64, 64, 32),
]

CUSTOM_INPUTS = [
    ("identity", torch.bfloat16, 13, 2560, 8, 8, 4),
]


def brute_force():
    act_fns = ["identity"]
    dtypes = [torch.bfloat16]
    bs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16)
    cs = (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        32,
        64,
        128,
        256,
        512,
        13,
        140,
        125,
        961,
        160,
        320,
        640,
        960,
        1280,
        1600,
        1920,
        2240,
        2560,
        1303,
        2602,
        3909,
    )
    rs = (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        17,
        8,
        16,
        64,
        128,
        256,
        512,
        1024,
    )
    gs = (
        1,
        2,
        3,
        4,
        8,
        16,
        32,
    )
    all_params = itertools.product(act_fns, dtypes, bs, cs, rs, rs, gs)

    err_inputs = filter(config_filter, all_params)
    err_inputs = filter(lambda x: x[4] == x[5], err_inputs)  # only allow inputs where H = W to reduce search space
    return err_inputs


def test_inputs(inputs, upcast_errors=True):
    err_inputs = []
    for params in tqdm(list(inputs)):
        err_params = check_params(params)
        if err_params:
            err_inputs.append(params)

    if not upcast_errors:
        return err_inputs

    # retry the error inputs but with a higher precision to see if a large error was because of precision issues
    # (or because of programmer error)
    for upcast_dtype in [torch.float, torch.double]:
        inputs = err_inputs[:]
        err_inputs = []
        for params in tqdm(inputs):
            act_fn, dtype, b, c, h, w, g = params
            if (
                torch.finfo(dtype).resolution <= torch.finfo(upcast_dtype).resolution
            ):  # only bother checking with the upcasted dtype if it has higher precision to the error-ed test case
                err_inputs.append(params)
                continue
            params = (act_fn, upcast_dtype, b, c, h, w, g)
            err_params = check_params(params)
            if err_params:
                err_inputs.append(params)

    return err_inputs


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False, edgeitems=1)
    inputs = CUSTOM_INPUTS
    # inputs = brute_force()
    err_inputs = test_inputs(inputs)

    if len(err_inputs) > 0:
        print(red("Error inputs found, writing to err_inputs.txt"))
        with open("err_inputs.txt", "w") as f:
            f.write("[\n")
            f.write(",\n".join(map(str, err_inputs)))
            f.write("\n]")
        f.close()
    else:
        print(green("No errors found :)"))
