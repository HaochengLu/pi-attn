"""CUDA-graph timing helper shared by the GDN decode benchmarks."""
import torch
from torch.profiler import profile, ProfilerActivity


def graph_time(fn, iters=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): fn()
    for _ in range(10): gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): gr.replay()
    e1.record(); torch.cuda.synchronize()
    t = e0.elapsed_time(e1) / iters * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(50): gr.replay()
        torch.cuda.synchronize()
    kern = sum(e.device_time_total for e in p.key_averages() if e.device_type.name == 'CUDA') / 50
    return t, kern
