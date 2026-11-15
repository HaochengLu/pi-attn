"""mma.sync operand symmetry on this GPU (see sym_probe.cu). Regimes: Gaussian at several magnitudes, wide
exponents, raw random bf16 / f32 bit patterns (NaN, inf, subnormals included), and injected special values
(+-0, +-inf, NaN, subnormal, largest / smallest normal) at 2-30% density. Single steps and chains of 2 and 8."""
import os, sys, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name="sym_probe_r5", sources=[os.path.join(HERE, "sym_probe.cu")],
           extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"])
print(torch.cuda.get_device_name(0), f"sm_{major}{minor}", flush=True)
torch.manual_seed(0)
dev = "cuda"
SPEC16 = torch.tensor([0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x0001, 0x8001, 0x007f, 0x0080, 0x7f7f, 0xff7f, 0x3f80, 0xbf80],
                      dtype=torch.int32)
SPEC32 = torch.tensor([0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x00000001, 0x80000001,
                       0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff], dtype=torch.int64)

def raw16(shape):
    return torch.randint(0, 65536, shape, device=dev, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
def raw32(shape):
    return torch.randint(0, 1 << 32, shape, device=dev, dtype=torch.int64).to(torch.int32).view(torch.float32)
def inject16(t, p):
    m = torch.rand(t.shape, device=dev) < p
    s = SPEC16.to(dev)[torch.randint(0, len(SPEC16), t.shape, device=dev)].to(torch.int16).view(torch.bfloat16)
    return torch.where(m, s, t)
def inject32(t, p):
    m = torch.rand(t.shape, device=dev) < p
    s = SPEC32.to(dev)[torch.randint(0, len(SPEC32), t.shape, device=dev)].to(torch.int32).view(torch.float32)
    return torch.where(m, s, t)

def gauss(n, steps, sx, sc, wide):
    ex = torch.randint(-wide, wide + 1, (n, steps, 16, 16), device=dev).float() if wide else torch.zeros(n, steps, 16, 16, device=dev)
    x = (torch.randn(n, steps, 16, 16, device=dev) * sx * torch.exp2(ex)).bfloat16()
    y = (torch.randn(n, steps, 8, 16, device=dev) * sx * torch.exp2(ex[:, :, :8])).bfloat16()
    c = torch.randn(n, 16, 8, device=dev) * sc
    return x, y, c

REGIMES = {
    "gauss 1/1": lambda n, s: gauss(n, s, 1, 1, 0),
    "gauss 3/100": lambda n, s: gauss(n, s, 3, 100, 0),
    "gauss 1/1e-3": lambda n, s: gauss(n, s, 1, 1e-3, 0),
    "gauss c=0": lambda n, s: gauss(n, s, 1, 0, 0),
    "wide 2^+-6": lambda n, s: gauss(n, s, 1, 1, 6),
    "wide 2^+-20, c 1e4": lambda n, s: gauss(n, s, 30, 1e4, 20),
    "wide 2^+-60 (overflow/underflow)": lambda n, s: gauss(n, s, 1, 1, 60),
    "raw bits": lambda n, s: (raw16((n, s, 16, 16)), raw16((n, s, 8, 16)), raw32((n, 16, 8))),
    "gauss + 2% specials": lambda n, s: tuple(f(t, 0.02) for f, t in zip((inject16, inject16, inject32), gauss(n, s, 1, 1, 0))),
    "gauss + 30% specials": lambda n, s: tuple(f(t, 0.30) for f, t in zip((inject16, inject16, inject32), gauss(n, s, 1, 1, 4))),
    "softmax-like (P in [0,1], V gauss)": lambda n, s: (torch.rand(n, s, 16, 16, device=dev).pow(8).bfloat16(), torch.randn(n, s, 8, 16, device=dev).bfloat16(), torch.randn(n, 16, 8, device=dev) * 3),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
gbad = gtot = gnan = 0
for steps in (1, 2, 8):
    for name, make in REGIMES.items():
        bad = tot = nanboth = nans = 0
        for rep in range(4):
            x, y, c = (t.contiguous() for t in make(N, steps))
            o0 = torch.zeros(N, 16, 8, device=dev); o1 = torch.full((N, 16, 8), 7.0, device=dev)
            ext.sym(x, y, c, o0, o1, steps); torch.cuda.synchronize()
            b0, b1 = o0.view(torch.int32), o1.view(torch.int32)
            ne = b0 != b1
            bad += ne.sum().item(); tot += ne.numel()
            nanboth += (ne & o0.isnan() & o1.isnan()).sum().item(); nans += o0.isnan().sum().item()
        gbad += bad; gtot += tot; gnan += nanboth
        print(f"SYM steps={steps} {name}: {bad} of {tot} elements differ (of which both NaN, payload only: {nanboth}); NaN outputs {nans}", flush=True)
print(f"SYM TOTAL: {gbad} of {gtot} differ ({gnan} NaN-payload-only)", flush=True)
