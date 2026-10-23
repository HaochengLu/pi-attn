"""Probe 16b: one burst of 64 launches over DRAM-resident weight copies, launched eagerly from
Python (as kernels/gemm_burst.py did before) and captured in a CUDA graph (as the engine
runs decode), for cuBLAS and every PI candidate. Usage (from kernels/): python ../archive/ceiling_probes/graph_mix/eager_vs_graph.py"""
import sys,torch
sys.path.insert(0,'.')
import test_gemm as G
from gemm_burst import POOL


def burst(fn, Ws, X, n=64, reps=5):  # eager, as gemm_burst.py before probe 16
    ts = []
    for _ in range(reps + 1):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(n):
            fn(X, Ws[i % len(Ws)])
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / n)
    return sorted(ts[1:])[reps // 2]
def gburst(fn,Ws,X,n=64):
    fn(X,Ws[0]); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n): fn(X,Ws[i%len(Ws)])
    ts=[]
    for _ in range(6):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e)*1000/n)
    return sorted(ts[1:])[2]
for name in ['0.6B o','0.6B down','0.6B gateup','8B down','8B gateup']:
    N,K=G.SHAPES[name]; copies=max(4,POOL//(N*K*2))
    Ws=[(torch.randn(N,K,device='cuda')*0.02).bfloat16() for _ in range(copies)]
    X=torch.randn(1,K,device='cuda').bfloat16()
    cb=lambda x,w: torch.matmul(x,w.T)
    ks=[k for k in G.candidates(1,N,K)]
    pe={k:burst(lambda x,w:G.ext.linear(x,w,k,False),Ws,X) for k in ks}
    pg={k:gburst(lambda x,w:G.ext.linear(x,w,k,False),Ws,X) for k in ks}
    be=min(pe,key=pe.get); bg=min(pg,key=pg.get)
    print(f"{name:10s} cublas eager {burst(cb,Ws,X):6.2f} graph {gburst(cb,Ws,X):6.2f} | pi best eager k{be} {pe[be]:6.2f} (graph {pg[be]:6.2f}) | pi best graph k{bg} {pg[bg]:6.2f}",flush=True)
    del Ws; torch.cuda.empty_cache()
