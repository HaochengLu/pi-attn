"""Probe 16c: segment-split ids 54/55 against the balanced (30/31) and segment-parallel (48)
kernels and cuBLAS on the two S = 2 shapes of Qwen3-0.6B, graph-captured bursts; checks
bitwise equality with id 31 first. Usage (from kernels/): python ../archive/ceiling_probes/graph_mix/segsplit.py"""
import sys,torch
sys.path.insert(0,'.')
import test_gemm as G
from gemm_burst import burst, POOL
for M in (1, 8, 16):
    for name in ['0.6B o','0.6B down']:
        N,K=G.SHAPES[name]; print(name, M, 'cands', [k for k in G.candidates(M,N,K) if k in (54,55)])
        copies=max(4,POOL//(N*K*2))
        Ws=[(torch.randn(N,K,device='cuda')*0.02).bfloat16() for _ in range(copies)]
        X=torch.randn(M,K,device='cuda').bfloat16()
        # bitwise vs id 31
        ref=G.ext.linear(X,Ws[0],31,False)
        for k in (54,55): assert torch.equal(G.ext.linear(X,Ws[0],k,False),ref), k
        r={'cublas':burst(lambda x,w: torch.matmul(x,w.T),Ws,X)}
        for k in (30,31,48,54,55): r[k]=burst(lambda x,w,k=k:G.ext.linear(x,w,k,False),Ws,X)
        print('  ',' '.join(f'{k}:{v:.2f}' for k,v in r.items()),flush=True)
        del Ws; torch.cuda.empty_cache()
