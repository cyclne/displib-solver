"""
Exact post-processing of a (sub)solution, conflict detection and event ordering.

AI-generated (Claude), October 2026.

Given the chosen paths and the chosen orientation of every conflict in the model, the
earliest feasible start times are a longest-path computation in the (generalised)
disjunctive graph. Doing this exactly (integers, no big-M tolerances) gives:
  * the schedule that is checked for new conflicts (paper: ConflictDetection),
  * the start-event order written to the DISPLIB solution file,
  * zero-cost cycles = swaps (paper Section 5.3, "No-swap constraints").
"""

import heapq
from collections import defaultdict

INF = float("inf")


class Interval:
    __slots__ = ("uid", "train", "res", "t_in", "t_out", "first", "runops", "never")

    def __init__(self, uid, train, res, t_in, t_out, first, runops, never):
        self.uid = uid
        self.train = train
        self.res = res
        self.t_in = t_in
        self.t_out = t_out
        self.first = first        # first op of the run on the path (reservation event)
        self.runops = runops      # [(k, next_k, rel_k)] for run ops k with a successor on the path
        self.never = never


def _scc(nodes, adj):
    """Iterative Tarjan. nodes: list of ids; adj: dict id -> list of (succ, w). Returns comp id per node."""
    index, low, onstack, comp = {}, {}, set(), {}
    stack, counter, ncomp = [], 0, 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        onstack.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for (w, _) in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    onstack.add(w)
                    work.append((w, iter(adj.get(w, ()))))
                    advanced = True
                    break
                elif w in onstack:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                while True:
                    w = stack.pop()
                    onstack.discard(w)
                    comp[w] = ncomp
                    if w == v:
                        break
                ncomp += 1
    return comp, ncomp


def path_index(inst, paths):
    """For every train: position of each op on its path."""
    return {i: {v: k for k, v in enumerate(p)} for i, p in paths.items()}


def earliest_times(inst, paths, orient):
    """
    Longest-path start times for given paths and conflict orientations.
    orient: dict key -> (first_uid, second_uid). Only pairs whose usages are both on the
    chosen paths are used. Returns dict (train, op) -> int time.
    Raises ValueError on a positive cycle (inconsistent input).
    """
    pos = path_index(inst, paths)
    adj = defaultdict(list)
    lbv = {}
    nodes = []
    for i, p in paths.items():
        tr = inst.trains[i]
        for k, v in enumerate(p):
            nodes.append((i, v))
            lbv[(i, v)] = tr.lb[v]
            if k + 1 < len(p):
                adj[(i, v)].append(((i, p[k + 1]), tr.dur[v]))
    for (first, second) in orient.values():
        uf, us = inst.usages[first], inst.usages[second]
        if uf.train not in paths or us.train not in paths:
            continue
        pf, ps = pos[uf.train], pos[us.train]
        if not any(v in pf for v in uf.ops) or not any(v in ps for v in us.ops):
            continue
        p_f = paths[uf.train]
        first_s = min((v for v in us.ops if v in ps), key=lambda v: ps[v])
        for v in uf.ops:
            if v in pf and pf[v] + 1 < len(p_f):
                nxt = p_f[pf[v] + 1]
                adj[(uf.train, nxt)].append(((us.train, first_s), inst.trains[uf.train].res[v][uf.resource]))
    comp, ncomp = _scc(nodes, adj)
    # condensation DAG in topological order: Tarjan returns SCCs in reverse topological order
    members = defaultdict(list)
    for v, c in comp.items():
        members[c].append(v)
    val = [-INF] * ncomp
    for c in range(ncomp):
        for v in members[c]:
            val[c] = max(val[c], lbv[v])
    incoming = defaultdict(list)
    for v, outs in adj.items():
        for (w, wt) in outs:
            if comp[v] != comp[w]:
                incoming[comp[w]].append((comp[v], wt))
            elif wt > 0:
                raise ValueError("positive cycle in disjunctive graph")
    for c in range(ncomp - 1, -1, -1):
        for (src, wt) in incoming[c]:
            val[c] = max(val[c], val[src] + wt)
    times = {}
    for v, c in comp.items():
        times[v] = int(val[c])
    return times


def intervals(inst, paths, times):
    """Exact occupation interval [t_in, t_out) of every usage on the chosen paths."""
    out = []
    for i, p in paths.items():
        tr = inst.trains[i]
        seen = set()
        for k, v in enumerate(p):
            for uid in tr.op_usages[v]:
                if uid in seen:
                    continue
                seen.add(uid)
                u = inst.usages[uid]
                runops, t_out, first = [], -INF, v
                kk = k
                never = False
                while kk < len(p) and p[kk] in u.ops:
                    op = p[kk]
                    rel = tr.res[op][u.resource]
                    if kk + 1 < len(p):
                        nxt = p[kk + 1]
                        runops.append((op, nxt, rel))
                        t_out = max(t_out, times[(i, nxt)] + rel)
                    else:
                        never = True
                    kk += 1
                if never:
                    t_out = INF
                out.append(Interval(uid, i, u.resource, times[(i, first)], t_out, first, runops, never))
    return out


def detect_conflicts(ivs):
    """All pairs of usages of different trains whose intervals overlap: s1 < e2 and s2 < e1."""
    byres = defaultdict(list)
    for iv in ivs:
        byres[iv.res].append(iv)
    found = {}
    for r, lst in byres.items():
        lst.sort(key=lambda x: (x.t_in, x.t_out))
        for a_idx, A in enumerate(lst):
            for B in lst[a_idx + 1:]:
                if B.t_in >= A.t_out:
                    break
                if B.train == A.train:
                    continue
                if A.t_in < B.t_out:
                    key = (min(A.uid, B.uid), max(A.uid, B.uid))
                    found[key] = (A, B)
    return found


def filter_dominated(found):
    """
    Paper Sec. 5.2: per train pair, drop a conflict C' if another conflict C'' of the same
    two trains contains both intervals of C'. Zero-length intervals are never dominated
    (they may be swap pivots). Conflicts are processed by resource name; the first wins ties.
    """
    bypair = defaultdict(list)
    for key, (A, B) in found.items():
        if A.train > B.train:
            A, B = B, A
        bypair[(A.train, B.train)].append((key, A, B))
    keep = {}
    for pair, lst in bypair.items():
        lst.sort(key=lambda e: (e[1].res, e[0]))
        kept = []
        for key, A, B in lst:
            dominated = False
            if A.t_in < A.t_out and B.t_in < B.t_out:
                for (_, A2, B2) in kept:
                    if (A2.t_in <= A.t_in and A.t_out <= A2.t_out and
                            B2.t_in <= B.t_in and B.t_out <= B2.t_out):
                        dominated = True
                        break
            if not dominated:
                kept.append((key, A, B))
                keep[key] = (A, B)
    return keep


def order_events(inst, paths, times, ivs, orient, max_cycles=50):
    """
    Produce the DISPLIB start-event sequence. Events are sorted by time; at equal times the
    order must respect "end of the earlier usage before start of the later usage".
    Returns (events, []) or (None, cycles) where each cycle is a list of arcs
    [(src, dst, info)] forming a zero-cost cycle (= a swap that DISPLIB forbids).
    Up to max_cycles cycles are collected in one call.
    """
    adj = defaultdict(list)
    indeg = defaultdict(int)
    nodes = []
    for i, p in paths.items():
        for k, v in enumerate(p):
            nodes.append((i, v))
            if k + 1 < len(p):
                adj[(i, v)].append(((i, p[k + 1]), ("train", i, v, p[k + 1])))
    # handover arcs at equal times
    byres = defaultdict(list)
    starts = defaultdict(lambda: defaultdict(list))
    for iv in ivs:
        byres[iv.res].append(iv)
        starts[iv.res][iv.t_in].append(iv)
    for r, lst in byres.items():
        for A in lst:
            if A.never:
                continue
            for (k, nxt, rel) in A.runops:
                if rel != 0:
                    continue
                T = times[(A.train, nxt)]
                for B in starts[r].get(T, ()):
                    if B.train == A.train:
                        continue
                    if not _before(A, B, orient):
                        continue
                    adj[(A.train, nxt)].append(((B.train, B.first), ("hand", A.uid, B.uid, A.train, k, nxt)))
    for v, outs in adj.items():
        for (w, _) in outs:
            indeg[w] += 1
    heap = []
    for v in nodes:
        if indeg[v] == 0:
            heapq.heappush(heap, (times[v], v))
    events, done = [], set()
    removed = set()          # handover arcs dropped to continue after a cycle was found
    cycles = []

    def drain():
        while heap:
            t, v = heapq.heappop(heap)
            events.append((t, v[0], v[1]))
            done.add(v)
            for (w, info) in adj.get(v, ()):
                if (v, w, info) in removed:
                    continue
                indeg[w] -= 1
                if indeg[w] == 0:
                    heapq.heappush(heap, (times[w], w))

    drain()
    while len(events) < len(nodes) and len(cycles) < max_cycles:
        # find a cycle among the remaining nodes by walking predecessors
        radj = defaultdict(list)
        for v, outs in adj.items():
            if v in done:
                continue
            for (w, info) in outs:
                if w not in done and (v, w, info) not in removed:
                    radj[w].append((v, info))
        start = next(v for v in nodes if v not in done)
        seen_at, walk, v = {}, [], start
        while v not in seen_at:
            seen_at[v] = len(walk)
            u, info = radj[v][0]
            walk.append((u, v, info))
            v = u
        cyc = walk[seen_at[v]:]
        cycles.append(cyc)
        # drop one handover arc of this cycle and continue, to collect more cycles per round
        arc = next(a for a in cyc if a[2][0] == "hand")
        removed.add(arc)
        indeg[arc[1]] -= 1
        if indeg[arc[1]] == 0:
            heapq.heappush(heap, (times[arc[1]], arc[1]))
        drain()
    if not cycles:
        return events, []
    return None, cycles


def _before(A, B, orient):
    """Is usage A (strictly or by tie-break) ordered before usage B?"""
    if A.t_out <= B.t_in and B.t_out <= A.t_in:
        key = (min(A.uid, B.uid), max(A.uid, B.uid))
        o = orient.get(key)
        if o is not None:
            return o[0] == A.uid
        return A.train < B.train
    return A.t_out <= B.t_in


def objective_value(inst, paths, times):
    """Exactly as the official verifier computes it."""
    total = 0
    for (i, v), c in inst.obj.items():
        if (i, v) not in times:
            continue
        t = times[(i, v)]
        total += c.coeff * max(0, t - c.threshold)
        total += c.increment * (1 if t >= c.threshold else 0)
    return total
