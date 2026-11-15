import re,gzip,collections,sys
def load(f):
    ins=[]
    for l in gzip.open(f,'rt'):
        m=re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(.*?);',l)
        if m: ins.append((int(m.group(1),16), m.group(2).strip()))
    return ins
for f in sys.argv[1:]:
    ins=load(f)
    idx={a:i for i,(a,t) in enumerate(ins)}
    loops=[]
    for a,t in ins:
        m=re.search(r'\bBRA\b.*?0x([0-9a-f]+)',t)
        if m and int(m.group(1),16)<a and (a-int(m.group(1),16))//16>1500: loops.append((int(m.group(1),16),a))
    mx=max(int(x) for a,t in ins for x in re.findall(r'\bR(\d+)',t))
    print(f, 'total', len(ins), 'maxreg', mx)
    for lo_a,hi_a in loops:
        body=ins[idx[lo_a]:idx[hi_a]+1]
        h=0; start=0; steps=[]
        for k,(a,t) in enumerate(body):
            if 'HMMA' in t: h+=1
            if h==64 and re.search(r'@P\d BRA',t):
                steps.append((start,k)); start=None; h=0
            if start is None and 'BAR.SYNC' in t: start=k-2
        for (s0,e0) in steps:
            if s0 is None: continue
            c=collections.Counter()
            for a,t in body[s0:e0]:
                t2=re.sub(r'^@!?P\d\s+','',t); c[t2.split()[0].split('.')[0]]+=1
            first_ldsm=[k for k,(a,t) in enumerate(body[s0:e0]) if 'LDSM' in t][0]
            print(f'  loop@{lo_a:x} len {(hi_a-lo_a)//16}: step {e0-s0} instrs, before first LDSM {first_ldsm};', sorted(c.items(), key=lambda x:-x[1])[:24])
