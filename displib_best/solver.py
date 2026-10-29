"""
Algorithm 1 of Stärk, Borndörfer & Lindner (ATMOS 2026): conflict-driven decomposition.

AI-generated (Claude), October 2026.

  1: build the disjunctive graph from the train DAGs           (instance.py, model.py)
  2: C <- {}                                                    (active conflict set)
  3: S <- {{i} : i in trains}                                   (one subproblem per train)
  4: repeat
  5-7:  solve every changed subproblem S given C                (SubModel.solve)
  8:    C_new <- ConflictDetection                              (timing.detect_conflicts)
  9-11: if C_new is empty: return                               (+ no-swap check, see below)
 12:    C <- C + C_new                                          (+ order-agreement cuts (21))
 13:    merge the subproblems linked by C_new

Extras from Section 5.3: domination filter in conflict detection, order-agreement cuts,
no-swap cuts (added when a swap cycle is observed), and the heuristic configuration
(time limit per iteration, fixing the most stable decisions after some time).
"""

import math
import time
from collections import defaultdict

from .model import SubModel, get_backend
from .timing import (earliest_times, intervals, detect_conflicts, filter_dominated,
                     order_events, objective_value)
from .cuts import CutFactory

DEFAULTS = dict(
    time_limit=600.0,      # total seconds
    iter_limit=None,       # seconds per subproblem solve (paper heuristic: 20)
    gap=1e-4,              # MIPGap per subproblem (0 for proofs)
    threads=0,
    decompose=True,        # Algorithm 1 decomposition (False: one model with all trains)
    domination=True,       # drop dominated conflicts in detection
    agreement=True,        # order-agreement cuts (21)
    route_cuts=True,       # route-conditional agreement cuts (proactive two-train no-swap cuts)
    heuristic=False,       # fix stable decisions after fix_after seconds
    fix_after=50.0,
    fix_streak=4,
    fix_frac=0.05,
    repair=True,           # if time runs out: greedy repair to still return a feasible schedule
    fixed_routes=False,    # keep every train on its best stand-alone route (round 1)
    two_step=False,        # with fixed_routes: after convergence, free the routes and continue
    verbose=True,
    verbose_gurobi=False,
    backend="gurobi",      # or "highs" (no licence needed, slower)
)


class Solver:
    def __init__(self, inst, **kw):
        self.inst = inst
        self.p = dict(DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if v is not None})
        self.gp, self.GRB = get_backend(self.p["backend"])
        if self.p["backend"] == "gurobi":
            env = self.gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            self.env = env
        else:
            self.env = None
        n = len(inst.trains)
        if self.p["decompose"]:
            self.parent = list(range(n))
            self.members = {i: {i} for i in range(n)}
        else:
            self.parent = [0] * n
            self.members = {0: set(range(n))}
        self.models, self.model_members = {}, {}
        self.C = set()
        self.conf_by_train = defaultdict(set)
        self.conf_by_pair = defaultdict(set)
        self.cuts = {}                  # cid -> (terms, rhs, trains)
        self.cuts_by_train = defaultdict(set)
        self.paths, self.orient = {}, {}
        self.group_res, self.group_times = {}, {}
        self.fixed, self.streak = {}, {}
        self.ever_fixed = False
        self.route_fix = None        # paths that y is fixed to (fixed-route mode)
        self.phase1 = None
        self.cf = CutFactory(inst, route_conditional=self.p["route_cuts"])
        self.log = []
        self.warnings = []

    # ------------------------------------------------------------ union-find over trains
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        r, o = min(ra, rb), max(ra, rb)
        self.parent[o] = r
        self.members[r] |= self.members.pop(o)
        for d in (self.models, self.model_members, self.group_res, self.group_times):
            d.pop(o, None)
        return r

    # ------------------------------------------------------------ bookkeeping
    def _key_trains(self, key):
        return self.inst.usages[key[0]].train, self.inst.usages[key[1]].train

    def _add_conflict(self, key):
        if key in self.C:
            return None
        self.C.add(key)
        ti, tj = self._key_trains(key)
        self.conf_by_train[ti].add(key)
        self.conf_by_train[tj].add(key)
        pair = (min(ti, tj), max(ti, tj))
        new_cuts = []
        if self.p["agreement"]:
            for k2 in self.conf_by_pair[pair]:
                new_cuts += self.cf.agreement_cuts(key, k2)
        self.conf_by_pair[pair].add(key)
        root = self.union(ti, tj) if self.p["decompose"] else 0
        for cid, terms, rhs, trains in new_cuts:
            self._add_cut(cid, terms, rhs, trains)
        return root

    def _add_cut(self, cid, terms, rhs, trains):
        if cid in self.cuts:
            return
        self.cuts[cid] = (terms, rhs, frozenset(trains))
        for i in trains:
            self.cuts_by_train[i].add(cid)

    def _group_conflicts(self, mem):
        keys = set()
        for i in mem:
            keys |= self.conf_by_train[i]
        return [k for k in keys if all(t in mem for t in self._key_trains(k))]

    def _group_cuts(self, mem):
        cids = set()
        for i in mem:
            cids |= self.cuts_by_train[i]
        return [c for c in cids if self.cuts[c][2] <= mem]

    def _sync_model(self, root):
        """Build (or update incrementally) the model of a group."""
        mem = frozenset(self.members[root])
        mdl = self.models.get(root)
        params = {"gap": self.p["gap"], "threads": self.p["threads"],
                  "verbose_gurobi": self.p["verbose_gurobi"], "backend": self.p["backend"]}
        if mdl is None or self.model_members.get(root) != mem:
            mdl = SubModel(self.inst, mem, params, env=self.env)
            self.models[root] = mdl
            self.model_members[root] = mem
        for key in self._group_conflicts(mem):
            mdl.add_conflict(key)
        for cid in self._group_cuts(mem):
            terms, rhs, _ = self.cuts[cid]
            mdl.add_cut(cid, terms, rhs)
        for key, (f, s) in self.fixed.items():
            if key in mdl.conflicts:
                mdl.fix(f, s, 1)
        if self.route_fix is not None:
            mdl.fix_routes(self.route_fix)
        return mdl

    # ------------------------------------------------------------ one subproblem
    def _solve_group(self, root, tl):
        mdl = self._sync_model(root)
        mem = self.members[root]
        mdl.set_start({i: self.paths[i] for i in mem if i in self.paths}, self.orient)
        res = mdl.solve(tl)
        if res["status"] == self.GRB.INFEASIBLE:
            fixed_here = [k for k in self.fixed if k in mdl.conflicts]
            if fixed_here:
                for k in fixed_here:
                    del self.fixed[k]
                mdl.unfix_all()
                res = mdl.solve(tl)
            if res["status"] == self.GRB.INFEASIBLE:
                raise RuntimeError(f"subproblem of trains {sorted(mem)} is infeasible")
        self.group_res[root] = res
        if res["paths"] is None:
            return False
        for i in mem:
            self.paths[i] = res["paths"][i]
        for key in mdl.conflicts:
            if key in res["orient"]:
                self.orient[key] = res["orient"][key]
            else:
                self.orient.pop(key, None)
        gp_paths = {i: self.paths[i] for i in mem}
        gp_orient = {k: self.orient[k] for k in mdl.conflicts if k in self.orient}
        self.group_times[root] = earliest_times(self.inst, gp_paths, gp_orient)
        # Self-check: for the same routes and orders, the exact earliest schedule can only be
        # cheaper than the model's optimum. If it is cheaper at proven optimality, the model is
        # over-constrained (a modelling bug) and the reported lower bound would be wrong.
        exact = objective_value(self.inst, gp_paths, self.group_times[root])
        if res["status"] == self.GRB.OPTIMAL and exact < res["obj"] - 1e-6 and self.p["gap"] <= 1e-9:
            self.warnings.append(f"group {sorted(mem)}: model obj {res['obj']} > exact {exact}")
            if self.p["verbose"]:
                print("WARNING:", self.warnings[-1], flush=True)
        return True

    # ------------------------------------------------------------ main loop
    def run(self):
        inst, p = self.inst, self.p
        t0 = time.time()
        dirty = set(self.members)
        rounds = 0
        result = None
        while True:
            rounds += 1
            ok = True
            for root in sorted(dirty, key=lambda r: len(self.members[r])):
                remaining = p["time_limit"] - (time.time() - t0)
                if remaining <= 0:
                    ok = False
                    break
                tl = remaining if p["iter_limit"] is None else min(remaining, p["iter_limit"])
                solved = self._solve_group(root, tl)
                # no solution within the per-iteration limit: keep trying while time is left
                while not solved:
                    remaining = p["time_limit"] - (time.time() - t0)
                    if remaining <= 1:
                        break
                    tl = min(remaining, 2 * tl)
                    solved = self._solve_group(root, tl)
                if not solved:
                    ok = False
                    break
            if not ok:
                result = self._finish(t0, rounds, timeout=True)
                break
            dirty = set()
            times = {}
            for root in self.members:
                times.update(self.group_times[root])
            ivs = intervals(inst, self.paths, times)
            found = detect_conflicts(ivs)
            bad = [k for k in found if k in self.C]
            new = {k: v for k, v in found.items() if k not in self.C}
            events, cycles = order_events(inst, self.paths, times, ivs, self.orient)
            n_found = len(new)
            if p["domination"] and new:
                new = filter_dominated(new)
            self._log_round(rounds, t0, n_found, len(new), bad, cycles)
            if rounds == 1 and p["fixed_routes"] and self.route_fix is None:
                self.route_fix = {i: list(pth) for i, pth in self.paths.items()}
            if not new and not bad and not cycles:
                if self.route_fix is not None and p["two_step"]:
                    obj1 = objective_value(inst, self.paths, times)
                    self.phase1 = dict(objective=obj1, time=round(time.time() - t0, 2), rounds=rounds)
                    if p["verbose"]:
                        print(f"fixed-route phase converged: objective {obj1}; freeing the routes", flush=True)
                    self.route_fix = None
                    for mdl in self.models.values():
                        mdl.unfix_routes()
                    dirty = set(self.members)
                    continue
                result = self._finish(t0, rounds, timeout=False, times=times, events=events)
                break
            if bad and not new and not cycles:
                raise RuntimeError(f"conflicts already in C still overlap: {bad[:5]}")
            touched = set()
            for key in sorted(new):
                r = self._add_conflict(key)
                touched.add(self.find(self._key_trains(key)[0]))
            for cycle in cycles:
                cid, terms, rhs, trains, need = self.cf.noswap_cut(cycle, self.paths)
                for key in need:
                    self._add_conflict(key)
                self._add_cut(cid, terms, rhs, trains)
                touched.add(self.find(next(iter(trains))))
            dirty = {self.find(r) for r in touched}
            self._update_streaks_and_fix(t0)
            if time.time() - t0 >= p["time_limit"]:
                result = self._finish(t0, rounds, timeout=True)
                break
        result["log"] = self.log
        result["warnings"] = self.warnings
        return result

    # ------------------------------------------------------------ heuristic fixing
    def _update_streaks_and_fix(self, t0):
        for key in self.C:
            o = self.orient.get(key)
            if o is None:
                continue
            last = self.streak.get(key)
            self.streak[key] = (o, last[1] + 1) if last and last[0] == o else (o, 1)
        p = self.p
        if not p["heuristic"] or time.time() - t0 < p["fix_after"]:
            return
        unfixed = [k for k in self.C if k not in self.fixed and k in self.orient]
        if not unfixed:
            return
        cand = [k for k in unfixed if self.streak[k][1] >= p["fix_streak"]]
        cand.sort(key=lambda k: -self.streak[k][1])
        nfix = math.ceil(p["fix_frac"] * len(unfixed))
        for k in cand[:nfix]:
            self.fixed[k] = self.orient[k]
            self.ever_fixed = True
            root = self.find(self._key_trains(k)[0])
            mdl = self.models.get(root)
            if mdl is not None and k in mdl.conflicts:
                mdl.fix(self.orient[k][0], self.orient[k][1], 1)

    # ------------------------------------------------------------ logging / results
    def _log_round(self, rounds, t0, n_found, n_new, bad, cycles):
        sizes = [len(m) for m in self.members.values()]
        ub = sum(r["obj"] for r in self.group_res.values() if r.get("obj") is not None)
        lb = sum(r["bound"] for r in self.group_res.values() if r.get("bound") is not None)
        entry = dict(round=rounds, time=round(time.time() - t0, 2), groups=len(sizes),
                     max_group=max(sizes), conflicts=len(self.C), detected=n_found, added=n_new,
                     cuts=len(self.cuts), swaps=len(cycles), bad=len(bad),
                     model_obj=round(ub, 2), model_bound=round(lb, 2), fixed=len(self.fixed))
        self.log.append(entry)
        if self.p["verbose"]:
            print(f"[{entry['time']:8.2f}s] round {rounds:3d} | groups {entry['groups']:4d} "
                  f"(max {entry['max_group']:4d}) | C {entry['conflicts']:6d} | new {n_new:5d}"
                  f"{' (of ' + str(n_found) + ')' if n_found != n_new else ''} | cuts {entry['cuts']:5d}"
                  f"{' | swaps ' + str(len(cycles)) if cycles else ''} | relaxation obj {ub:.1f} bound {lb:.1f}"
                  f"{' | fixed ' + str(len(self.fixed)) if self.fixed else ''}", flush=True)

    def _finish(self, t0, rounds, timeout, times=None, events=None):
        inst = self.inst
        status = None
        if not timeout:
            obj = objective_value(inst, self.paths, times)
            all_opt = all(r["status"] == self.GRB.OPTIMAL for r in self.group_res.values())
            lb = None if self.ever_fixed else sum(r["bound"] for r in self.group_res.values())
            status = "optimal" if (all_opt and not self.ever_fixed) else "feasible"
            if self.route_fix is not None:
                status, lb = "feasible (fixed routes)", None
        else:
            obj, lb = None, None
            if self.p["repair"] and len(self.paths) == len(inst.trains):
                rep = repair(inst, self.paths, self.orient)
                if rep is not None:
                    times, events = rep
                    obj = objective_value(inst, self.paths, times)
                    status = "feasible (repaired after timeout)"
            if obj is None:
                status = "no feasible solution (timeout)"
            if not self.ever_fixed and len(self.group_res) == len(self.members):
                bounds = [r["bound"] for r in self.group_res.values()]
                if all(b is not None for b in bounds):
                    lb = sum(bounds)
        sol = None
        if events is not None:
            sol = {"objective_value": int(obj),
                   "events": [{"time": int(t), "train": i, "operation": v} for (t, i, v) in events]}
        sizes = [len(m) for m in self.members.values()]
        return dict(status=status, objective=obj, lower_bound=lb, time=round(time.time() - t0, 2),
                    phase1=self.phase1,
                    rounds=rounds, conflicts=len(self.C), cuts=len(self.cuts),
                    agreement_cuts=sum(1 for c in self.cuts if c[0] == "agree"),
                    route_cuts=sum(1 for c in self.cuts if c[0] == "agree_r"),
                    swap_cuts=sum(1 for c in self.cuts if c[0] == "swap"),
                    max_group=max(sizes), groups=len(sizes), solution=sol)


def repair(inst, paths, orient, max_iter=100000):
    """
    Greedy fallback used only when time runs out: keep the routes, orient every remaining
    overlap first-come-first-served (earlier reservation first), recompute exact times,
    repeat. Returns (times, events) or None (deadlock / swap).
    """
    o = dict(orient)
    flipped = set()
    for _ in range(max_iter):
        try:
            times = earliest_times(inst, paths, o)
        except ValueError:
            return None
        ivs = intervals(inst, paths, times)
        found = detect_conflicts(ivs)
        if not found:
            events, cycles = order_events(inst, paths, times, ivs, o)
            if not cycles:
                return times, events
            # a swap at equal times: reverse one handover of each cycle (delays one train)
            progress = False
            for cyc in cycles:
                for (_, _, info) in cyc:
                    if info[0] != "hand":
                        continue
                    ua, ub = info[1], info[2]
                    key = (min(ua, ub), max(ua, ub))
                    if key in flipped or inst.usages[ub].never_released:
                        continue
                    o[key] = (ub, ua)
                    flipped.add(key)
                    progress = True
                    break
            if not progress:
                return None
            continue
        for key, (A, B) in found.items():
            first, second = (A, B) if (A.t_in, A.train) <= (B.t_in, B.train) else (B, A)
            if first.never:
                first, second = second, first
            o[key] = (first.uid, second.uid)
    return None
