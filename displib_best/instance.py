"""
Instance preprocessing for the Stärk-style DISPLIB solver.

AI-generated (Claude), October 2026. Read and verified module by module before use.

What this module computes, per train (paper notation in brackets):
  * the operation DAG with predecessors/successors              [G^i = (V^i, A^i)]
  * levels = topological depth (longest path from the entry)     [l(v), V^i_l]
  * level start-time bounds, including levels that a path skips  [alpha^i_l, beta^i_l]
  * earliest possible start of each operation (EST, no resources)
  * number of entry->v and v->exit paths, used to decide which
    operations / resource usages are on every path ("mandatory")
  * resource usages = runs of consecutive operations that hold
    the same resource (connected components of the resource
    subgraph), with their reserving operations and release edges [rho, R^in_rho, R^out_rho]

DISPLIB semantics used here (from the official spec + verifier):
  * an operation ends when the train starts its next operation;
  * a resource used by operation k is free for other trains only
    `release_time` after k has ended;
  * resources of the exit operation are never released;
  * objective: c*max(0, t-threshold) + d*[t >= threshold], 0 if the
    operation is not on the train's path.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field

INF = float("inf")


@dataclass
class ObjComp:
    train: int
    op: int
    threshold: int
    coeff: int
    increment: int


@dataclass
class Usage:
    """One resource usage rho: a train holds `resource` over a run of operations."""
    uid: int
    train: int
    resource: str
    ops: frozenset                 # operations of the run (a convex component)
    rin: list                      # reserving ops: run ops entered from outside (or the entry op)
    rin_by_level: dict             # level -> [reserving ops on that level]
    entering_edges: list           # (p, v) with p outside, v inside; x_rho = sum of their y (+1 if entry inside)
    entry_inside: bool
    release_by_level: dict         # level of head u -> [(v, u, release_time_of_v)]
    release_cover: dict            # level -> True if every path through the run uses exactly one edge of that group
    never_released: bool           # run contains the exit operation
    mandatory: bool                # every path of the train visits the run
    est_in: float                  # lower bound on the reservation time
    est_out: float                 # lower bound on the release time


@dataclass
class Train:
    idx: int
    n: int
    lb: list
    ub: list
    dur: list
    res: list                      # per op: {resource: release_time}
    succ: list
    pred: list
    level: list = field(default_factory=list)
    L: int = 0
    level_ops: list = field(default_factory=list)
    level_lb: list = field(default_factory=list)
    level_ub: list = field(default_factory=list)
    est: list = field(default_factory=list)
    pin: list = field(default_factory=list)
    pout: list = field(default_factory=list)
    npaths: int = 0
    mandatory: list = field(default_factory=list)
    reach: list = field(default_factory=list)   # bitset per op: ops reachable from it (incl. itself)
    usages: list = field(default_factory=list)  # uids
    op_usages: list = field(default_factory=list)  # per op: list of uids


class Instance:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.name = path
        self.trains = []
        for i, tr in enumerate(raw["trains"]):
            n = len(tr)
            lb, ub, dur, res, succ = [], [], [], [], []
            for op in tr:
                lb.append(int(op.get("start_lb", 0)))
                ub.append(op.get("start_ub", INF))
                dur.append(int(op.get("min_duration", 0)))
                r = {}
                for u in op.get("resources", []):
                    name = u["resource"]
                    r[name] = max(r.get(name, 0), int(u.get("release_time", 0)))
                res.append(r)
                succ.append(list(op.get("successors", [])))
            pred = [[] for _ in range(n)]
            for v in range(n):
                for s in succ[v]:
                    if s <= v:
                        raise ValueError(f"train {i}: operations are not topologically ordered")
                    pred[s].append(v)
            self.trains.append(Train(i, n, lb, ub, dur, res, succ, pred))

        # Objective: like the official verifier, one component per (train, op), the last one wins.
        self.obj = {}
        for c in raw.get("objective", []):
            if c.get("type", "op_delay") != "op_delay":
                raise ValueError("only op_delay objective components are supported")
            key = (c["train"], c["operation"])
            self.obj[key] = ObjComp(c["train"], c["operation"], int(c.get("threshold", 0)),
                                    int(c.get("coeff", 0)), int(c.get("increment", 0)))

        for t in self.trains:
            self._levels_and_bounds(t)

        # Global time horizon. All time variables are bounded by T_MAX, which makes every
        # big-M below well defined. It is generous (sum of all durations and release times
        # on top of the latest earliest-start), so it does not cut off reasonable schedules.
        max_est = max(max(t.est) for t in self.trains)
        sum_dur = sum(sum(t.dur) for t in self.trains)
        sum_rel = sum(max(r.values(), default=0) for t in self.trains for r in t.res)
        self.max_rel = max((max(r.values(), default=0) for t in self.trains for r in t.res), default=0)
        self.T_MAX = max_est + sum_dur + sum_rel + 1
        for t in self.trains:
            t.level_ub = [min(u, self.T_MAX) for u in t.level_ub]

        self.usages = []
        self.usages_by_resource = defaultdict(list)
        for t in self.trains:
            self._build_usages(t)

    # ------------------------------------------------------------------ levels
    def _levels_and_bounds(self, t):
        n = t.n
        lvl = [0] * n
        for v in range(n):
            for s in t.succ[v]:
                lvl[s] = max(lvl[s], lvl[v] + 1)
        t.level = lvl
        t.L = lvl[n - 1]
        t.level_ops = [[] for _ in range(t.L + 1)]
        for v in range(n):
            t.level_ops[lvl[v]].append(v)

        # Ops whose start time can equal the time variable of level l:
        # the ops on level l, plus heads b of arcs (a, b) that jump over level l
        # (a skipped level's time is squeezed between its neighbours, see model.py).
        cand = [list(ops) for ops in t.level_ops]
        for a in range(n):
            for b in t.succ[a]:
                for l in range(lvl[a] + 1, lvl[b]):
                    cand[l].append(b)
        t.level_lb = [min(t.lb[v] for v in c) for c in cand]
        t.level_ub = [max(t.ub[v] for v in c) for c in cand]

        # Earliest start (lower bound, ignores other trains).
        est = [INF] * n
        est[0] = t.lb[0]
        for v in range(n):
            est[v] = max(est[v], t.lb[v])
            for s in t.succ[v]:
                est[s] = min(est[s], est[v] + t.dur[v]) if est[s] != INF else est[v] + t.dur[v]
        # the min above keeps the earliest arrival; enforce lbs once more in order
        for v in range(n):
            est[v] = max(est[v], t.lb[v])
        t.est = est

        pin = [0] * n
        pin[0] = 1
        for v in range(n):
            for s in t.succ[v]:
                pin[s] += pin[v]
        pout = [0] * n
        pout[n - 1] = 1
        for v in range(n - 2, -1, -1):
            pout[v] = sum(pout[s] for s in t.succ[v])
        t.pin, t.pout, t.npaths = pin, pout, pout[0]
        t.mandatory = [pin[v] * pout[v] == t.npaths for v in range(n)]

        reach = [0] * n
        for v in range(n - 1, -1, -1):
            r = 1 << v
            for s in t.succ[v]:
                r |= reach[s]
            reach[v] = r
        t.reach = reach

    # ------------------------------------------------------------------ usages
    def _new_usage(self, t, r, ops):
        S = frozenset(ops)
        rin, entering = [], []
        entry_inside = 0 in S
        for v in sorted(S):
            outside_preds = [p for p in t.pred[v] if p not in S]
            if v == 0 or outside_preds:
                rin.append(v)
            for p in outside_preds:
                entering.append((p, v))
        rin_by_level = defaultdict(list)
        for v in rin:
            rin_by_level[t.level[v]].append(v)

        exit_op = t.n - 1
        never = exit_op in S
        rel_by_level = defaultdict(list)
        for v in S:
            rel = t.res[v][r]
            for u in t.succ[v]:
                # Edge inside the run with rel_v <= dur_u is implied by later edges of the run.
                if u in S and rel <= t.dur[u]:
                    continue
                rel_by_level[t.level[u]].append((v, u, rel))

        through = t.pout[0] if entry_inside else 0
        for (p, v) in entering:
            through += t.pin[p] * t.pout[v]
        mandatory = through == t.npaths
        cover = {}
        for l, edges in rel_by_level.items():
            cov = sum(t.pin[v] * t.pout[u] for (v, u, _) in edges)
            cover[l] = (cov == through)

        est_in = min(t.est[v] for v in rin) if rin else INF
        if never:
            est_out = INF
        else:
            est_out = min((t.est[u] + rel for es in rel_by_level.values() for (v, u, rel) in es), default=INF)
        u = Usage(len(self.usages), t.idx, r, S, rin, dict(rin_by_level), entering, entry_inside,
                  dict(rel_by_level), cover, never, mandatory, est_in, est_out)
        self.usages.append(u)
        self.usages_by_resource[r].append(u.uid)
        t.usages.append(u.uid)
        for v in S:
            t.op_usages[v].append(u.uid)

    def _build_usages(self, t):
        t.op_usages = [[] for _ in range(t.n)]
        byres = defaultdict(list)
        for v in range(t.n):
            for r in t.res[v]:
                byres[r].append(v)
        for r, vs in byres.items():
            S = set(vs)
            parent = {v: v for v in vs}

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for v in vs:
                for s in t.succ[v]:
                    if s in S:
                        parent[find(v)] = find(s)
            comps = defaultdict(list)
            for v in vs:
                comps[find(v)].append(v)
            for comp in comps.values():
                if self._convex(t, comp):
                    self._new_usage(t, r, comp)
                else:
                    # A path could leave the run and come back: merging would be inexact,
                    # so fall back to one usage per operation (exact, just more conflicts).
                    for v in comp:
                        self._new_usage(t, r, [v])

    @staticmethod
    def _convex(t, comp):
        mask = 0
        for v in comp:
            mask |= 1 << v
        out = 0
        for v in comp:
            for s in t.succ[v]:
                if not (mask >> s) & 1:
                    out |= t.reach[s]
        out &= ~mask
        while out:
            low = out & -out
            w = low.bit_length() - 1
            out ^= low
            if t.reach[w] & mask:
                return False
        return True

    # ------------------------------------------------------------------ stats
    def summary(self):
        return {
            "trains": len(self.trains),
            "ops": sum(t.n for t in self.trains),
            "edges": sum(len(s) for t in self.trains for s in t.succ),
            "usages": len(self.usages),
            "levels": sum(t.L + 1 for t in self.trains),
            "T_MAX": self.T_MAX,
        }
