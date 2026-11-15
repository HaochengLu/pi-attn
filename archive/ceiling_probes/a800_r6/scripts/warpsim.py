"""In-order single-warp timing model of one loop of a kernel's SASS: an instruction issues no earlier than
ISSUE cycles after the previous one and no earlier than its operands are ready (latency by kind). Reports
cycles per step in steady state and where the waits are. usage: warpsim.py file.sass.gz [loop index] [issue]"""
import re, gzip, sys, collections
LAT = {'HMMA': 31, 'LDSM': 31, 'SHFL': 26, 'MUFU': 19, 'F2FP': 6, 'FMNMX': 5, 'LDS': 26, 'LDG': 120, 'BAR': 30, 'LDGSTS': 4}
def lat(op):
    return LAT.get(op.split('.')[0], 4)
def load(f):
    ins = []
    for l in gzip.open(f, 'rt'):
        m = re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(.*?);', l)
        if m: ins.append((int(m.group(1), 16), m.group(2).strip()))
    return ins
def regs(tok):
    out = []
    for m in re.finditer(r'\b(R|P|UR)(\d+)(\.64)?', tok):
        k = m.group(1); n = int(m.group(2))
        out.append((k, n))
        if m.group(3): out.append((k, n + 1))
    return out
def decode(t):
    pred = None
    m = re.match(r'@(!?)(P\d)\s+(.*)', t)
    if m: pred = ('P', int(m.group(2)[1:])); t = m.group(3)
    op, _, rest = t.partition(' ')
    ops = [x.strip() for x in re.split(r',(?![^\[]*\])', rest)] if rest else []
    base = op.split('.')[0]
    dst, src = [], []
    if pred: src.append(pred)
    if base == 'HMMA':
        d = regs(ops[0])[0][1]; a = regs(ops[1])[0][1]; b = regs(ops[2])[0][1]
        dst = [('R', d + i) for i in range(4)]
        src += [('R', a + i) for i in range(4)] + [('R', b), ('R', b + 1)]
        c = regs(ops[3]); 
        if c: src += [('R', c[0][1] + i) for i in range(4)]
    elif base == 'LDSM':
        d = regs(ops[0])[0][1]; dst = [('R', d + i) for i in range(4)]; src += regs(ops[1])
    elif base in ('LDGSTS', 'STS', 'STG', 'BAR', 'DEPBAR', 'LDGDEPBAR', 'BRA', 'BSSY', 'BSYNC', 'NOP', 'RET', 'EXIT'):
        for o in ops: src += regs(o)
    else:
        if ops:
            d0 = regs(ops[0]) if not ops[0].startswith('[') else []
            if d0 and ops[0] not in ('PT',):
                dst = d0[:1]
                if 'WIDE' in op or op.startswith('LDG') and '64' in op: dst = [d0[0], (d0[0][0], d0[0][1] + 1)]
                rest_ops = ops[1:]
            else: rest_ops = ops[1:] if ops[0] == 'PT' else ops
            for o in rest_ops: src += regs(o)
            if base in ('ISETP', 'FSETP') and len(ops) > 1 and ops[1] != 'PT' and regs(ops[1]) and ops[1].startswith('P'):
                dst.append(regs(ops[1])[0])
        if pred and dst: src += dst  # a predicated write keeps the old value
    return base, dst, src
def main():
    ins = load(sys.argv[1]); idx = {a: i for i, (a, t) in enumerate(ins)}
    li = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ISSUE = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    loops = []
    for a, t in ins:
        m = re.search(r'\bBRA\b.*?0x([0-9a-f]+)', t)
        if m and int(m.group(1), 16) < a and (a - int(m.group(1), 16)) // 16 > 1500: loops.append((int(m.group(1), 16), a))
    lo, hi = loops[li]
    body = ins[idx[lo]:idx[hi] + 1]
    # the two steps: from BAR.SYNC to the far predicated BRA after the 64th HMMA; non-copier path: skip from the
    # instruction after BAR to the last BSYNC before the first LDSM
    steps = []; h = 0; start = None
    for k, (a, t) in enumerate(body):
        if 'BAR.SYNC' in t and start is None: start = k
        if 'HMMA' in t: h += 1
        if h == 64 and re.search(r'@P\d BRA', t): steps.append((start, k)); start = None; h = 0
    seq = []
    for s0, e0 in steps:
        blk = body[s0:e0]
        first_ldsm = [k for k, (a, t) in enumerate(blk) if 'LDSM' in t][0]
        last_bsync = max([k for k, (a, t) in enumerate(blk[:first_ldsm]) if 'BSYNC' in t] + [0])
        seq.append([blk[0]] + blk[last_bsync + 1:])
    ready = collections.defaultdict(float); t = 0.0
    wait_by = collections.Counter(); per_step = []
    for rep in range(6):
        for si, st in enumerate(seq):
            t0 = t; waits = collections.Counter()
            for a, txt in st:
                base, dst, src = decode(txt)
                want = t + ISSUE
                rdy = max([ready[s] for s in src] + [0.0])
                if rdy > want:
                    # blame the producer kind
                    blame = max(src, key=lambda s: ready[s]); waits[prod.get(blame, '?')] += rdy - want
                issue = max(want, rdy)
                for d in dst: ready[d] = issue + lat(txt.split()[-1] if False else (re.sub(r'^@!?P\d\s+', '', txt)).split()[0]); prod[d] = base
                t = issue
            per_step.append((len(st), t - t0, waits))
    for n, c, w in per_step[-2:]:
        print(f'  step: {n} instrs (non-copier path), {c:.0f} cycles = {n * ISSUE:.0f} issue + {c - n * ISSUE:.0f} wait; waits by producer: {dict((k, round(v)) for k, v in w.most_common(8))}')
prod = {}
print(sys.argv[1], 'loop', sys.argv[2] if len(sys.argv) > 2 else 0)
main()
