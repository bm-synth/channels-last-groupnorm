# Exhaustive search to check if GN outputs are equal to reference.
from gnnhwc import GN_NHWC
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, datetime, time, os, itertools, sys

class GN_Naive(nn.Module):
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
        N, C, H, W = x.shape
        self.x = x
        xr = x.view(N, self.G, H*W*C//self.G)
        means = xr.mean(dim=2, keepdim=True)
        rstds = torch.rsqrt(xr.var(dim=2, correction=0, keepdim=True) + self.eps)
        self.means = means[:, :, 0]
        self.rstds = rstds[:, :, 0]
        xnorm = (xr - means) * rstds
        self.xnorm = xnorm
        xnormr = xnorm.reshape(N, C, H, W) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return xnormr

    def bwd(self, dy): # bwd pass for debugging
        N, C, H, W = dy.shape
        G = self.G
        D = C // G
        dyr = dy.view(N, G, D, H, W)
        xr = x.view(N, G, D, H, W)
        dy_sum = dyr.sum((3, 4))
        xdy_sum = (dyr * xr).sum((3, 4))
        dy_gamma = (self.weight.view(1, G, D) * dy_sum).sum(2)
        xdy_gamma = (self.weight.view(1, G, D) * xdy_sum).sum(2)
        dweight = ((xdy_sum - self.means[:,:,None] * dy_sum) * self.rstds[:,:,None]).sum(0)
        dbias = dy_sum.sum(0)
        c1 = (self.means * dy_gamma - xdy_gamma) / (H*W*D) * self.rstds**3
        c2 = -self.means * c1 - dy_gamma*self.rstds / (H*W*D)
        dx = self.weight.view(1,G,D,1,1)*self.rstds.view(N,G,1,1,1)*dyr + c1.view(N,G,1,1,1)*xr+c2.view(N,G,1,1,1)

def red(text): return '\033[91m' + str(text) + '\033[0m'
def green(text): return '\033[92m' + str(text) + '\033[0m'
def yellow(text): return '\033[93m' + str(text) + '\033[0m'
def blue(text): return '\033[94m' + str(text) + '\033[0m'

def get_act_fn(act_str):
    return {
        'identity': lambda x: x,
        'silu': F.silu,
        'relu': F.relu,
        'gelu': F.gelu,
        'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
    }[act_str]

def config_filter(x): # returns true if config is valid
    ACT_FN, DTYPE, B, C, H, W, G = x
    if C % G != 0:
        return False
    if H * W == 1: # this causes an autograd problem where it gets confused since the tensor is both contiguous in NCHW/NHWC format 
        return False

    dtype_size = torch.finfo(DTYPE).bits / 8
    estimated_mem_usage_gib = (25 * dtype_size * B * C * H * W) / 2**30 #  this is just a rough estimate, likely wrong
    if estimated_mem_usage_gib > 3: # vram filter
        return False
    return True

ERRS = {
    torch.bfloat16: 1e-6,
    torch.float16: 1e-7,
    torch.float: 1e-10,
    torch.double: 1e-20,
}

bigx = None
def check_params(params, verbose=True):
    global bigx
    if bigx is None:
        bigx = torch.randn(128*1024*1024)
    vprint = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None
    ACT_FN, DTYPE, B, C, H, W, G = params
    vprint(blue(f'output testing | ACT_FN: {ACT_FN} | DTYPE: {DTYPE} |B: {B:<2} | C: {C:<4} | H: {H:<4} | W: {W:<4} | G: {G:<3}'))
    xc = bigx[:B*C*H*W].reshape((B, C, H, W)).to(DTYPE).cuda()
    x = xc.to(memory_format=torch.channels_last)
    xc.requires_grad_(True)
    x.requires_grad_(True)
    torch.random.manual_seed(0)

    gn2 = GN_NHWC(G, C, activation=ACT_FN).cuda().to(DTYPE)
    #gn2 = nn.GroupNorm(G, C).cuda().to(DTYPE)

    gn1 = nn.GroupNorm(G, C).cuda().to(DTYPE)
    act_fn = get_act_fn(ACT_FN)
    gnref = GN_Naive(G, C).cuda().to(DTYPE)
    with torch.no_grad(): # copy weights
        w = torch.randn((C,), dtype=DTYPE)
        b = torch.randn((C,), dtype=DTYPE)
        gn1.weight.copy_(w.detach().float())
        gn1.bias.copy_(b.detach().float())
        gn2.weight.copy_(w.detach())
        gn2.bias.copy_(b.detach())
        gnref.weight.copy_(w.detach())
        gnref.bias.copy_(b.detach())

    gn_layers = [gn1, gn2, gnref]
    g1 = act_fn(gn1(xc))
    g2 = gn2(x)
    rand_dy = torch.rand_like(g2)
    rand_dy /= rand_dy.numel() ** 0.5 # to prevent false positive errors from ocurring because of really large magnitude losses

    err_params = False
    gref = gref_dx = None
    def vprint_err(x_ref, x_test, x_naive_fn, bwd=True, left_pad=0):
        nonlocal gref, gref_dx
        lpad = ' ' * left_pad
        with torch.no_grad():
            err = F.mse_loss(x_ref, x_test)

        if err < ERRS[DTYPE]:
            vprint(green(f'{lpad}Negligible difference (err: {err:.2e}) found'))
        else:
            if gref is None:
                gref = act_fn(gnref(xc))
            if bwd and gref_dx is None:
                xc.grad = None
                gref.backward(rand_dy)
                gref_dx = xc.grad.clone()

            with torch.no_grad():
                x_naive = x_naive_fn()
                err_ref_naive = F.mse_loss(x_ref, x_naive)
                err_test_naive = F.mse_loss(x_test, x_naive)

            if err_test_naive < err_ref_naive:
                vprint(yellow(f'{lpad}Negligible difference (err: {err:.2e}, test-naive: {err_test_naive:.2e}, ref-naive: {err_ref_naive:.2e}) found'))
            else:
                vprint(red(f'{lpad}Error: {err:.2e}, test-naive: {err_test_naive:.2e}, ref-naive: {err_ref_naive:.2e}'))
                return True
        return False

    vprint('  FORWARD')
    err_params = vprint_err(g1, g2, lambda: gref, bwd=False, left_pad=4) or err_params
    vprint('  BACKWARD')
    xc.grad = None
    g1.backward(rand_dy)
    g1_dx = xc.grad.clone()

    x.grad = None
    g2.backward(rand_dy)
    g2_dx = x.grad.clone()

    vprint('    wrt X')
    err_params = vprint_err(g1_dx, g2_dx, lambda: gref_dx, left_pad=6) or err_params
    vprint('    wrt weight')
    err_params = vprint_err(gn1.weight.grad, gn2.weight.grad, lambda: gnref.weight.grad, left_pad=6) or err_params
    vprint('    wrt bias')
    err_params = vprint_err(gn1.bias.grad, gn2.bias.grad, lambda: gnref.bias.grad, left_pad=6) or err_params

    return err_params

CUSTOM_INPUTS = [
    ('identity', torch.double, 1,    1,   2,   2,    1),
    ('identity', torch.double, 2, 3909,   5,   5,    3),
    ('identity', torch.double, 1, 2062,   5,   5, 1031),
    ('identity', torch.double, 3, 4096,   7,   7,    4),
    ('identity', torch.double, 1, 4096,   7,   7,    4),
    ('identity', torch.double, 2,  160,   8,   8,  160),
    ('identity', torch.double, 1,    3,   7,   7,    1),
    ('identity', torch.double, 1,    1,   4,   4,    1),
    ('identity', torch.double, 1,  128,   8,   8,    8),
    ('identity', torch.double, 2, 1280,   8,   8,   32),
    ('identity', torch.double, 2,  640,  16,  16,   32),
    ('identity', torch.double, 2, 2560,   8,   8,   32),
    ('identity', torch.double, 2, 1280,  16,  16,   32),
    ('identity', torch.double, 2,  320,  32,  32,   32),
    ('identity', torch.double, 2, 1920,  16,  16,   32),
    ('identity', torch.double, 2, 2560,  16,  16,   32),
    ('identity', torch.double, 2,  640,  32,  32,   32),
    ('identity', torch.double, 2,  960,  32,  32,   32),
    ('identity', torch.double, 2, 1280,  32,  32,   32),
    ('identity', torch.double, 2,  320,  64,  64,   32),
    ('identity', torch.double, 2, 1920,  32,  32,   32),
    ('identity', torch.double, 2,  640,  64,  64,   32),
    ('identity', torch.double, 8,  128,  64,  64,   32),
]

def brute_force():
    ACT_FNS = ['identity']
    DTYPES = [torch.float]
    Bs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16)
    Cs = (
        1, 2, 3, 4, 5, 6, 7, 32, 64, 128, 256, 512,
        13, 140, 125, 961,
        160, 320, 640, 960, 1280, 1600, 1920, 2240, 2560, 1303, 2602, 3909
    )
    Rs = (
        2, 3, 4, 5, 6, 7, 8, 9, 10, 17,
        8, 16, 64, 128, 256, 512,
        1024,
    )
    Gs = (1, 2, 3, 4, 8, 16, 32,)
    all_params = itertools.product(ACT_FNS, DTYPES, Bs, Cs, Rs, Rs, Gs)
    inputs = None

    err_inputs = filter(config_filter, all_params)
    err_inputs = filter(lambda x: x[4] == x[5], err_inputs) # only allow inputs where H = W to reduce search space
    return err_inputs

def test_inputs(inputs, upcast_errors=True):
    err_inputs = []
    for params in tqdm(list(inputs)):
        err_params = check_params(params)
        if err_params:
            err_inputs.append(params)
    
    if not upcast_errors:
        return err_inputs

    # retry the error inputs but with a higher precision to see if a large error was because of precision issues (or because of programmer error)
    for UPCAST_DTYPE in [torch.float, torch.double]:
        inputs = err_inputs[:]
        err_inputs = []
        for params in tqdm(inputs):
            ACT_FN, DTYPE, B, C, H, W, G = params
            if torch.finfo(DTYPE).resolution <= torch.finfo(UPCAST_DTYPE).resolution: # only bother checking with the upcasted dtype if it has higher precision to the error-ed test case
                err_inputs.append(params)
                continue
            params = (ACT_FN, UPCAST_DTYPE, B, C, H, W, G)
            err_params = check_params(params)
            if err_params:
                err_inputs.append(params)

    return err_inputs

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False, edgeitems=1)
    inputs = CUSTOM_INPUTS
    inputs = brute_force()
    err_inputs = test_inputs(inputs)

    if len(err_inputs) > 0:
        print(red('Error inputs found:'))
        print(err_inputs)
    else:
        print(green('No errors found :)'))
