"""Driver for gx.cu: identity hashes, spec check, M-invariance, accuracy vs fp64, speed vs cuBLAS.
usage: python drv.py --tasks ident,acc,speed [--quick]
"""
import argparse, ctypes, hashlib, subprocess, time
import numpy as np
import torch

dev = "cuda"
BF = torch.bfloat16
SHAPES = [(4096, 2048), (2048, 6144), (12288, 2048)]  # (N, K)


def build():
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0]}{cap[1]}"
    subprocess.check_call(["nvcc", "-O3", "-std=c++17", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC",
                           "-o", "libgx.so", "gx.cu"])
    lib = ctypes.CDLL("./libgx.so")
    V, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    lib.gx_slice.argtypes = [V, I, I, I, I, I, V, V, V]
    lib.gx_gemm.argtypes = [I, I, V, L, V, L, V, V, V, I, I, I, I, I]
    lib.gx_rate.argtypes = [I, I, I, V]
    return lib, arch


def P(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else None


# name: mode, fmt_x, fmt_w, B, s_x, s_w   (mode 0 with fmt None = native chain on raw inputs)
VARS = {
    "I4": (4, 0, 0, 256, 0, 0),
    "I3": (3, 1, 1, 256, 0, 0),
    "I2": (2, 0, 2, 256, 0, 0),
    "I1": (1, 2, 2, 256, 0, 0),
    "K3": (5, 4, 4, 256, 0, 0),
    "K3b64": (5, 4, 4, 64, 0, 0),
    "F8": (0, 3, 3, 128, 0, 0),     # 8+8+7 = 23 bits
    "F12": (0, 3, 3, 16, 1, 2),     # 9+10+4 = 23 bits, fold every MMA
    "Fp24a": (0, 3, 3, 256, 0, 0),  # probe: 24 bits
    "Fp24b": (0, 3, 3, 16, 2, 2),   # probe: 24 bits
    "Fp25": (0, 3, 3, 16, 2, 3),    # probe: 25 bits
    "Fp27": (0, 3, 3, 16, 3, 4),    # probe: 27 bits (must be inexact)
    "N": (0, None, None, 0, 0, 0),
}
for b in (64, 1024):
    VARS[f"I4b{b}"] = (4, 0, 0, b, 0, 0)
    VARS[f"I3b{b}"] = (3, 1, 1, b, 0, 0)
VARS["I4b32"] = (4, 0, 0, 32, 0, 0)


_BUF = {}
DEFCFG = {}  # mode -> large-M cfg, filled by task_speed's first pass or left at 0


def _buf(key, shape, dtype):
    k = (key, tuple(shape), dtype)
    if k not in _BUF:
        _BUF[k] = torch.empty(shape, dtype=dtype, device=dev)
    return _BUF[k]


def slice_dev(lib, x, B, fmt, s=0, cache=None):
    """cache: reuse output buffers (timing loops only; results alias between calls)."""
    R, K = x.shape
    assert x.is_contiguous()
    if fmt is None:
        return x, None
    if cache is not None and fmt != 3:
        npl = 1 if fmt == 2 else 2
        pl, sc = _buf(cache, (npl, R, K), torch.uint8), _buf(cache, (R, K // B), torch.float32)
        rc = lib.gx_slice(P(x), R, K, B, fmt, 0, P(pl[0]), P(pl[1]) if npl == 2 else None, P(sc))
        assert rc == 0, rc
        return pl, sc
    if fmt == 3:
        out = torch.empty_like(x) if cache is None else _buf(cache, x.shape, BF)
        rc = lib.gx_slice(P(x), R, K, B, 3, s, P(out), None, None)
        assert rc == 0, rc
        return out, None
    npl = 1 if fmt == 2 else 2
    pl = torch.empty((npl, R, K), dtype=torch.uint8, device=dev)
    sc = torch.empty((R, K // B), dtype=torch.float32, device=dev)
    rc = lib.gx_slice(P(x), R, K, B, fmt, 0, P(pl[0]), P(pl[1]) if npl == 2 else None, P(sc))
    assert rc == 0, rc
    return pl, sc


def gemm_dev(lib, var, xp, wp, M, N, K, cfg=None, out=None, out32=False):
    mode, fx, fw, B, _, _ = VARS[var]
    if cfg is None:
        cfg = 1 if M <= 16 else DEFCFG.get(mode, 0)
    if out is None:
        out = torch.empty((M, N), dtype=torch.float32 if out32 else BF, device=dev)
    if mode == 0:
        RB, bsteps = 2 * K, (B // 16 if fx is not None else K // 16)
        rc = lib.gx_gemm(0, cfg, P(xp[0]), 0, P(wp[0]), 0, None, None, P(out), M, N, RB, bsteps, int(out32))
    else:
        rc = lib.gx_gemm(mode, cfg, P(xp[0]), M * K, P(wp[0]), N * K, P(xp[1]), P(wp[1]), P(out), M, N, K,
                         B // 32, int(out32))
    assert rc == 0, rc
    return out


def prep_w(lib, var, W):
    mode, fx, fw, B, sx, sw = VARS[var]
    return slice_dev(lib, W, B, fw, sw)


def run(lib, var, X, wp, cfg=None, out=None, out32=False, cache=None):
    mode, fx, fw, B, sx, sw = VARS[var]
    M, K = X.shape
    xp = slice_dev(lib, X, B, fx, sx, cache)
    N = wp[0].shape[-2]
    return gemm_dev(lib, var, xp, wp, M, N, K, cfg, out, out32)


# ------------------------------------------------------------------ spec (independent, torch) ----
def pow2d(e):  # exact 2^e as float64, built from bits
    return ((e.long() + 1023) << 52).view(torch.float64)


def dec(x):
    h = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (h >> 7) & 0xFF
    mm = h & 0x7F
    mm = torch.where(e > 0, mm | 0x80, mm)
    e = e.clamp(min=1)
    sg = 1 - 2 * ((h >> 15) & 1)
    return sg, mm, e


def spec_slice(x, B, fmt, s=0):
    R, K = x.shape
    sg, mm, e = dec(x)
    E = torch.where(mm > 0, e, torch.ones_like(e)).view(R, K // B, B).amax(-1)
    Ee = E.repeat_interleave(B, dim=1)
    if fmt == 3:
        Eq = (Ee - s).clamp(min=1)
        v = torch.round(mm.double() * pow2d(e - Eq))
        return sg.double() * v * pow2d(Eq - 134), None
    sh = {0: 7, 1: 6, 2: -1, 4: 5}[fmt]
    off = {0: 141, 1: 140, 2: 133, 4: 139}[fmt]
    v = torch.round(mm.double() * pow2d(sh - (Ee - e)))
    if fmt == 2:
        v = v.clamp(max=127)
    return sg.double() * v, pow2d(E - off).float()


def planes_to_v(pl, fmt):
    if fmt == 0:
        return pl[0].view(torch.int8).double() * 256 + pl[1].double()
    if fmt == 1:
        return pl[0].view(torch.int8).double() * 256 + pl[1].view(torch.int8).double()
    if fmt == 4:
        hi, lo = pl[0].view(torch.int8).double(), pl[1].view(torch.int8).double()
        assert float((hi - lo).abs().max()) <= 127
        return hi * 128 + lo
    return pl[0].view(torch.int8).double()


def check_slice(lib, x, B, fmt, s):
    got = slice_dev(lib, x, B, fmt, s)
    v, sc = spec_slice(x, B, fmt, s)
    if fmt == 3:
        return int((got[0].view(torch.int16) != v.float().to(BF).view(torch.int16)).sum())
    return int((planes_to_v(got[0], fmt) != v).sum()) + int((got[1] != sc).sum())


def spec_gemm(var, X, W, out32=False):
    mode, fx, fw, B, sx, sw = VARS[var]
    M, K = X.shape
    vx, scx = spec_slice(X, B, fx, sx)
    vw, scw = spec_slice(W, B, fw, sw)
    tot = torch.zeros((M, W.shape[0]), dtype=torch.float32, device=dev)
    for j in range(K // B):
        a, b = vx[:, j * B:(j + 1) * B], vw[:, j * B:(j + 1) * B]
        Pj = a @ b.T
        if mode == 3:
            la = torch.remainder(a + 128, 256) - 128
            lb = torch.remainder(b + 128, 256) - 128
            Pj = Pj - la @ lb.T
        f = Pj.float()
        if mode != 0:
            f = (f * scx[:, j, None]) * scw[None, :, j]
        tot = tot + f
    return tot if out32 else tot.to(BF)


# ------------------------------------------------------------------ data ----
def make(ds, M, N, K, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    X = rng.standard_normal((M, K), dtype=np.float32)
    W = rng.standard_normal((N, K), dtype=np.float32)
    if ds == "real":
        W *= np.float32(0.02)
        cols = rng.choice(K, 3, replace=False)
        X[:, cols] *= np.array([300.0, -1200.0, 2500.0], dtype=np.float32)
        mask = rng.random((M, K), dtype=np.float32) < 1e-3
        X[mask] *= np.float32(30.0)
    elif ds == "wide":
        X *= np.exp2(rng.integers(-12, 13, size=(M, K))).astype(np.float32)
        W *= np.exp2(rng.integers(-12, 13, size=(N, K))).astype(np.float32)
    elif ds == "gauss":
        W *= np.float32(0.02)
    elif ds == "sat":  # same-sign, near-maximal mantissas: every block partial is close to its worst case
        X = rng.integers(224, 256, size=(M, K)).astype(np.float32) / np.float32(128)
        X[7::8] *= np.sign(rng.standard_normal((M // 8, K), dtype=np.float32))
        X[:, 5::16] = np.float32(129.0 / 128 / 8)
        W = rng.integers(224, 256, size=(N, K)).astype(np.float32) / np.float32(128 * 64)
        W[:, 9::16] = np.float32(131.0 / 128 / 1024)
    elif ds == "cancel":
        W *= np.float32(0.02)
        W[:, 1::2] = -W[:, 0::2]
        pert = (rng.random((M, K // 2), dtype=np.float32) < 0.25) * rng.standard_normal((M, K // 2), dtype=np.float32)
        X[:, 1::2] = X[:, 0::2] * (1 + pert * np.float32(2.0 ** -5))
    Xt = torch.from_numpy(X).to(BF).contiguous()
    Wt = torch.from_numpy(W).to(BF).contiguous()
    return Xt.to(dev), Wt.to(dev), rng


def sha(t):
    return hashlib.sha256(t.contiguous().cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def neq(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def task_ident(lib, quick):
    names = ["I4", "K3", "I3", "I2", "I1", "I4b1024", "F8", "F12", "Fp24a", "Fp24b", "Fp25", "Fp27", "N"]
    M = 2048
    allh = hashlib.sha256()
    print("# ident: ds N K | variant hash | slice_bad spec_bad(16 rows) | Minv mismatches M=1,16,256,256(cfg1),256(cfg2),256(cfg3; -1 = smem too large on this GPU),256(cfg4),256(cfg5)")
    for ds in ("real", "wide", "cancel", "sat"):
        for si, (N, K) in enumerate(SHAPES):
            if quick and si > 0:
                continue
            X, W, rng = make(ds, M, N, K, 1000 + si)
            perm = torch.from_numpy(rng.permutation(M)).to(dev)
            print(f"IN {ds} {N} {K} x={sha(X)} w={sha(W)}", flush=True)
            print(f"ID {ds} {N} {K} cuBLAS {sha(X @ W.T)}")
            for v in names:
                mode, fx, fw, B, sx, sw = VARS[v]
                wp = prep_w(lib, v, W)
                Y = run(lib, v, X, wp)
                torch.cuda.synchronize()
                sb = sp = -1
                if fx is not None:
                    sb = check_slice(lib, X, B, fx, sx) + check_slice(lib, W, B, fw, sw)
                    sp = neq(spec_gemm(v, X[:16].contiguous(), W), Y[:16])
                mi = []
                for Ms, cfg in ((1, None), (16, None), (256, None), (256, 1), (256, 2), (256, 3), (256, 4), (256, 5)):
                    idx = perm[:Ms]
                    try:
                        Ys = run(lib, v, X[idx].contiguous(), wp, cfg=cfg)
                        mi.append(neq(Ys, Y[idx]))
                    except AssertionError:
                        mi.append(-1)
                h = sha(Y)
                if v in ("I4", "K3", "I3", "I2", "I1", "I4b1024", "F8", "F12", "Fp24b"):
                    allh.update(h.encode())
                print(f"ID {ds} {N} {K} {v} {h} | {sb} {sp} | {mi}", flush=True)
    print("ID_ALL_EXACT_VARIANTS", allh.hexdigest()[:16], flush=True)


def task_acc(lib, quick):
    names = ["cuBLAS", "cuBLAS@2048", "N", "I4", "I4b32", "I4b64", "I4b1024", "K3", "K3b64", "I3", "I3b64", "I2", "I1", "F8", "F12", "Fp24b"]
    M = 512
    print("# acc (M=512): ds N K variant | frac != correctly-rounded bf16 | median err (bf16 ulp of ref) | "
          "p99.9 err ulp | bf16-out rel l2 err | fp32-out rel l2 err")
    for ds in ("gauss", "real", "wide", "cancel"):
        for si, (N, K) in enumerate(SHAPES):
            if quick and si > 0:
                continue
            X, W, _ = make(ds, M, N, K, 1000 + si)
            ref = X.double() @ W.double().T
            m, e = torch.frexp(ref)
            ulp = pow2d(e - 8)
            cr = torch.round(m * 256) * ulp
            for v in names:
                if v == "cuBLAS":
                    Y, Y32 = X @ W.T, None
                elif v == "cuBLAS@2048":
                    Y, Y32 = (torch.cat([X, X, X, X]) @ W.T)[:M], None
                else:
                    wp = prep_w(lib, v, W)
                    Y = run(lib, v, X, wp)
                    Y32 = run(lib, v, X, wp, out32=True)
                yd = Y.double()
                err = (yd - ref).abs() / ulp
                err = torch.where(ref == 0, torch.zeros_like(err), err)
                rel = float((Y32.double() - ref).norm() / ref.norm()) if Y32 is not None else float("nan")
                relb = float((yd - ref).norm() / ref.norm())
                ef = err.flatten()[:8000000]
                print(f"ACC {ds} {N} {K} {v} | {float((yd != cr).double().mean()):.3e} | {float(ef.median()):.4f} | "
                      f"{float(ef.quantile(0.999)):.3f} | {relb:.3e} | {rel:.3e}", flush=True)
            del ref, cr, ulp


def bench(fn):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter(); fn(); torch.cuda.synchronize(); dt = time.perf_counter() - t
    n = int(min(300, max(3, 0.25 / max(dt, 1e-6))))
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / n)
    return best


def task_speed(lib, quick):
    names = ["N", "I1", "I2", "K3", "I3", "I4", "I4b1024", "F8", "F12"]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    sink = torch.zeros(sms * 4, device=dev)
    for ty, nm, kk in ((0, "bf16 m16n8k16", 16), (1, "s8 m16n8k32", 32)):
        for cps in (4, 2, 1):  # CTAs of 8 warps per SM
            it = 20000
            t = bench(lambda: lib.gx_rate(ty, it, sms * cps, P(sink)))
            nm_ = sms * cps * 8 * it * 8
            print(f"RATE {nm} {8 * cps} warps/SM: {nm_ / t / 1e9:.2f} G MMA/s = {nm_ * 16 * 8 * kk * 2 / t / 1e12:.1f} "
                  f"T(FL)OPS  ({sms} SMs)", flush=True)
    print("# speed: N K M | cuBLAS ms (TFLOPS) | per variant: total ms incl. X slicing at the best cfg (x cuBLAS time) best gemm-only over cfgs (x cuBLAS) [gemm-only ms cfg0|cfg2|cfg3|cfg4|cfg5]")
    for si, (N, K) in enumerate(SHAPES):
        X, W, _ = make("real", 2048, N, K, 1000 + si)
        Wt = W.T
        if quick and si > 0:
            continue
        for M in (1, 16, 256, 2048):
            Xm = X[:M].contiguous()
            Yc = torch.empty((M, N), dtype=BF, device=dev)
            tc = bench(lambda: torch.matmul(Xm, Wt, out=Yc))
            line = f"SP {N} {K} {M} | cuBLAS {tc * 1e3:.4f} ({2 * M * N * K / tc / 1e12:.1f})"
            for v in names:
                mode, fx, fw, B, sx, sw = VARS[v]
                wp = prep_w(lib, v, W)
                Y = torch.empty((M, N), dtype=BF, device=dev)
                tt = bench(lambda: run(lib, v, Xm, wp, out=Y, cache="x")) if M <= 16 else 0.0
                xp = slice_dev(lib, Xm, B, fx, sx)
                tg = bench(lambda: gemm_dev(lib, v, xp, wp, M, N, K, cfg=1 if M <= 16 else 0, out=Y))
                alt = []
                if M > 16:
                    for cf in (2, 3, 4, 5):
                        try:
                            alt.append(bench(lambda: gemm_dev(lib, v, xp, wp, M, N, K, cfg=cf, out=Y)))
                        except AssertionError:
                            alt.append(float("nan"))
                best = min([tg] + [a for a in alt if a == a])
                if M > 16:
                    cands = [tg] + alt
                    bi = min(range(5), key=lambda i: cands[i] if cands[i] == cands[i] else 1e9)
                    tt = bench(lambda: run(lib, v, Xm, wp, cfg=(0, 2, 3, 4, 5)[bi], out=Y, cache="x"))
                line += (f" | {v} {tt * 1e3:.4f} ({tt / tc:.2f}x) best-gemm {best * 1e3:.4f} ({best / tc:.2f}x) "
                         f"[{tg * 1e3:.4f}|" + "|".join(f"{a * 1e3:.4f}" for a in alt) + "]")
            print(line, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="ident")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    lib, arch = build()
    print("GPU", torch.cuda.get_device_name(0), arch, "torch", torch.__version__, flush=True)
    for t in a.tasks.split(","):
        t0 = time.time()
        {"ident": task_ident, "acc": task_acc, "speed": task_speed}[t](lib, a.quick)
        print(f"# task {t} done in {time.time() - t0:.0f}s", flush=True)
