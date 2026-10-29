"""
The compact MIP of Stärk, Borndörfer & Lindner (ATMOS 2026), Section 5.1, adapted to DISPLIB.
One SubModel = one subproblem S of Algorithm 1 (a group of trains plus the conflicts C
between them that have been observed so far).

AI-generated (Claude), October 2026.

Variables (paper notation):
  y[i,a,b]   in {0,1}  arc (a,b) of train i is on its path                      (y^i_{a,b})
  t[i,l]     >= 0      start time of level l of train i                          (t^i_l)
  x[i,a]     := sum of y into a (entry: 1)  -- an expression, not a variable    (eq. 2)
  t_in[r], t_out[r]    reservation / release time of resource usage r            (t^in_rho, t^out_rho)
  z[a,b]     in {0,1}  usage a is released before usage b is reserved            (z_{rho,sigma})
  w[k] >= 0, h[k] in {0,1}  delay cost and step (increment) indicator of objective component k

Constraints:
  flow (3)-(5); level durations (6); conditional start bounds (9);
  reservation upper bounds per level (12); release lower bounds per level (11);
  conflict orientation (13)-(14) and separation (15)-(16); objective (17)-(18).

DISPLIB adaptations (documented in README.md):
  * No no-wait arcs exist in DISPLIB, so (7) and (8) are not needed.
  * Release is anchored on the *edge* (v, u) leaving each run operation v:
    release >= start(u) + release_time(v). This is exact for DISPLIB, where an
    operation ends when the next one starts.
  * Objective uses the DISPLIB definition (start time of the op, step at t >= threshold).
"""

INF = float("inf")


def get_backend(name="gurobi"):
    """Return (gp-like module, GRB constants) for 'gurobi' (default) or 'highs'."""
    if name == "highs":
        from . import highs_gp
        return highs_gp, highs_gp.GRB
    import gurobipy
    return gurobipy, gurobipy.GRB


class SubModel:
    def __init__(self, inst, trains, params, env=None):
        self.inst = inst
        self.trains = sorted(trains)
        self.params = params
        self.gp, self.GRB = get_backend(params.get("backend", "gurobi"))
        gp = self.gp
        m = gp.Model("stark_sub", env=env) if env is not None else gp.Model("stark_sub")
        m.Params.OutputFlag = 1 if params.get("verbose_gurobi") else 0
        m.Params.IntFeasTol = 1e-9          # big-M constraints: avoid z = 0.99999 slack
        m.Params.MIPGap = params.get("gap", 1e-4)
        if params.get("threads"):
            m.Params.Threads = params["threads"]
        self.m = m
        self.y = {}
        self.t = {}
        self.x = {}
        self.tin = {}
        self.tout = {}
        self.tin_lb = {}
        self.tout_ub = {}
        self.xu = {}
        self.z = {}
        self.conflicts = set()
        self.cut_ids = set()
        self.obj_const = 0.0
        self._build_base()

    # ---------------------------------------------------------------- helpers
    def xop(self, i, v):
        return self.x[(i, v)]

    def _const(self, c):
        return self.gp.LinExpr(float(c))

    # ---------------------------------------------------------------- base model
    def _build_base(self):
        m, inst = self.m, self.inst
        obj = self.gp.LinExpr()
        for i in self.trains:
            tr = inst.trains[i]
            for v in range(tr.n):
                for s in tr.succ[v]:
                    self.y[(i, v, s)] = m.addVar(vtype=self.GRB.BINARY, name=f"y[{i},{v},{s}]")
            for l in range(tr.L + 1):
                self.t[(i, l)] = m.addVar(lb=tr.level_lb[l], ub=tr.level_ub[l], name=f"t[{i},{l}]")
            # x_a (eq. 2): written with the incoming arcs; the entry op is always selected.
            for v in range(tr.n):
                if v == 0:
                    self.x[(i, v)] = self._const(1.0)
                else:
                    self.x[(i, v)] = self.gp.quicksum(self.y[(i, p, v)] for p in tr.pred[v])
            # flow (3)-(5); (4) is implied for a DAG with a single sink
            if tr.succ[0]:   # a one-operation train (entry = exit) needs no flow
                m.addConstr(self.gp.quicksum(self.y[(i, 0, s)] for s in tr.succ[0]) == 1, name=f"src[{i}]")
            for v in range(1, tr.n - 1):
                m.addConstr(self.gp.quicksum(self.y[(i, p, v)] for p in tr.pred[v]) ==
                            self.gp.quicksum(self.y[(i, v, s)] for s in tr.succ[v]), name=f"flow[{i},{v}]")
            # level durations (6): t_{l+1} - t_l >= sum_{a in V_l} dur_a x_a
            for l in range(tr.L):
                rhs = self.gp.LinExpr()
                for a in tr.level_ops[l]:
                    if tr.dur[a] > 0:
                        rhs += tr.dur[a] * self.x[(i, a)]
                m.addConstr(self.t[(i, l + 1)] - self.t[(i, l)] >= rhs, name=f"dur[{i},{l}]")
            # conditional start bounds (9) and its upper-bound analogue
            for a in range(tr.n):
                l = tr.level[a]
                if tr.lb[a] > tr.level_lb[l]:
                    M = tr.lb[a] - tr.level_lb[l]
                    m.addConstr(self.t[(i, l)] >= tr.lb[a] - M * (1 - self.x[(i, a)]), name=f"lb[{i},{a}]")
                if tr.ub[a] < tr.level_ub[l]:
                    M = tr.level_ub[l] - tr.ub[a]
                    m.addConstr(self.t[(i, l)] <= tr.ub[a] + M * (1 - self.x[(i, a)]), name=f"ub[{i},{a}]")
        # objective (17)-(18), DISPLIB version
        for (i, v), c in inst.obj.items():
            if i not in self.trains:
                continue
            tr = inst.trains[i]
            l = tr.level[v]
            tl = self.t[(i, l)]
            xv = self.x[(i, v)]
            mand = tr.mandatory[v]
            if c.coeff > 0 and tr.level_ub[l] > c.threshold:
                w = m.addVar(lb=0.0, name=f"w[{i},{v}]")
                if mand:
                    m.addConstr(w >= c.coeff * (tl - c.threshold))
                else:
                    M = c.coeff * (tr.level_ub[l] - c.threshold)
                    m.addConstr(w >= c.coeff * (tl - c.threshold) - M * (1 - xv))
                obj += w
            if c.increment > 0 and tr.level_ub[l] >= c.threshold:
                if mand and tr.level_lb[l] >= c.threshold:
                    self.obj_const += c.increment
                else:
                    h = m.addVar(vtype=self.GRB.BINARY, name=f"h[{i},{v}]")
                    M = tr.level_ub[l] - (c.threshold - 1)
                    if mand:
                        m.addConstr(tl <= c.threshold - 1 + M * h)
                    else:
                        m.addConstr(tl <= c.threshold - 1 + M * h + M * (1 - xv))
                    obj += c.increment * h
        m.setObjective(obj + self.obj_const, self.GRB.MINIMIZE)

    # ---------------------------------------------------------------- usages (11), (12)
    def ensure_usage(self, uid):
        if uid in self.xu:
            return
        inst, m = self.inst, self.m
        u = inst.usages[uid]
        i = u.train
        tr = inst.trains[i]
        xr = self.gp.LinExpr(1.0) if u.entry_inside else self.gp.LinExpr()
        for (p, v) in u.entering_edges:
            xr += self.y[(i, p, v)]
        if u.mandatory:
            xr = self.gp.LinExpr(1.0)
        self.xu[uid] = xr

        # reservation time: t_in <= start of the reserving op (12)
        levels = sorted(u.rin_by_level)
        if len(levels) == 1:
            l = levels[0]
            lb = u.est_in if u.mandatory else tr.level_lb[l]
            tin = m.addVar(lb=lb, ub=inst.T_MAX, name=f"tin[{uid}]")
            m.addConstr(tin <= self.t[(i, l)])
        else:
            # optional run: t_in must stay feasible when the run is not visited at all
            lb = u.est_in if u.mandatory else min(tr.level_lb[k] for k in levels)
            tin = m.addVar(lb=lb, ub=inst.T_MAX, name=f"tin[{uid}]")
            for l in levels:
                # If the run is not entered on level l, the true reservation happens on a later
                # reserving level l' (t_in may need to be as large as t[l']) or on an earlier one
                # (then t_in <= t[l'] <= t[l] anyway). So M must cover t[l'] - t[l] for l' > l.
                later = [tr.level_ub[k] for k in levels if k > l]
                M = max(0.0, max(later) - tr.level_lb[l]) if later else 0.0
                sx = self.gp.quicksum(self.x[(i, v)] for v in u.rin_by_level[l])
                if M > 0:
                    m.addConstr(tin <= self.t[(i, l)] + M * (1 - sx))
                else:
                    m.addConstr(tin <= self.t[(i, l)])
        self.tin[uid] = tin
        self.tin_lb[uid] = lb

        # release time: t_out >= start(u) + release(v) for every edge (v,u) of the run (11)
        if u.never_released:
            self.tout[uid] = None
            self.tout_ub[uid] = INF
            return
        ub_out = max(tr.level_ub[l] + max(e[2] for e in es) for l, es in u.release_by_level.items())
        tout = m.addVar(lb=u.est_out, ub=ub_out, name=f"tout[{uid}]")
        for l, es in u.release_by_level.items():
            sy = self.gp.quicksum(self.y[(i, v, w)] for (v, w, _) in es)
            srel = self.gp.quicksum(rel * self.y[(i, v, w)] for (v, w, rel) in es if rel > 0)
            if u.mandatory and u.release_cover.get(l, False):
                m.addConstr(tout >= self.t[(i, l)] + srel)
            else:
                M = max(0.0, tr.level_ub[l] - u.est_out)
                m.addConstr(tout >= self.t[(i, l)] + srel - M * (1 - sy))
        self.tout[uid] = tout
        self.tout_ub[uid] = ub_out

    # ---------------------------------------------------------------- conflicts (13)-(16)
    def add_conflict(self, key):
        if key in self.conflicts:
            return
        a, b = key
        self.ensure_usage(a)
        self.ensure_usage(b)
        m, inst = self.m, self.inst
        ua, ub = inst.usages[a], inst.usages[b]
        zab = None if ua.never_released else m.addVar(vtype=self.GRB.BINARY, name=f"z[{a},{b}]")
        zba = None if ub.never_released else m.addVar(vtype=self.GRB.BINARY, name=f"z[{b},{a}]")
        xa, xb = self.xu[a], self.xu[b]
        if zab is None and zba is None:
            m.addConstr(xa + xb <= 1)
        else:
            S = self.gp.LinExpr()
            if zab is not None:
                S += zab
            if zba is not None:
                S += zba
            m.addConstr(S <= xa)
            m.addConstr(S <= xb)
            m.addConstr(xa + xb - 1 <= S)
            if zab is not None:
                M = self.tout_ub[a] - self.tin_lb[b]
                m.addConstr(self.tin[b] >= self.tout[a] - M * (1 - zab))
            if zba is not None:
                M = self.tout_ub[b] - self.tin_lb[a]
                m.addConstr(self.tin[a] >= self.tout[b] - M * (1 - zba))
        self.z[(a, b)] = zab
        self.z[(b, a)] = zba
        self.conflicts.add(key)

    def zvar(self, first, second):
        """Variable 'usage first released before usage second reserved', or None (constant 0)."""
        return self.z.get((first, second))

    # ---------------------------------------------------------------- cuts
    def add_cut(self, cid, terms, rhs):
        """sum(terms) <= rhs, terms: ('y', i, v, u) or ('z', first, second)."""
        if cid in self.cut_ids:
            return
        expr = self.gp.LinExpr()
        for t in terms:
            if t[0] == "y":
                expr += self.y[(t[1], t[2], t[3])]
            else:
                var = self.zvar(t[1], t[2])
                if var is not None:
                    expr += var
        self.m.addConstr(expr <= rhs, name=f"cut[{cid}]")
        self.cut_ids.add(cid)

    def fix(self, first, second, value):
        """Fix orientation of a conflict: value=1 means `first` before `second`."""
        v1 = self.zvar(first, second)
        v2 = self.zvar(second, first)
        if v1 is not None:
            v1.LB = v1.UB = value
        if v2 is not None:
            v2.LB = v2.UB = 1 - value

    def fix_routes(self, paths):
        """Fix y to the given paths (fixed-route mode: routing decided outside the MIP)."""
        for i in self.trains:
            on = set(zip(paths[i], paths[i][1:]))
            tr = self.inst.trains[i]
            for v in range(tr.n):
                for s in tr.succ[v]:
                    val = 1.0 if (v, s) in on else 0.0
                    var = self.y[(i, v, s)]
                    var.LB = var.UB = val

    def unfix_routes(self):
        for var in self.y.values():
            var.LB, var.UB = 0.0, 1.0

    def unfix_all(self):
        for var in self.z.values():
            if var is not None:
                var.LB, var.UB = 0.0, 1.0

    # ---------------------------------------------------------------- warm start
    def set_start(self, paths, orient):
        for (i, v, s), var in self.y.items():
            p = paths.get(i)
            if p is None:
                continue
            var.Start = 0.0
        for i, p in paths.items():
            if i not in self.trains:
                continue
            for a, b in zip(p, p[1:]):
                self.y[(i, a, b)].Start = 1.0
        for key in self.conflicts:
            o = orient.get(key)
            if o is None:
                continue
            f, s = o
            v1, v2 = self.zvar(f, s), self.zvar(s, f)
            if v1 is not None:
                v1.Start = 1.0
            if v2 is not None:
                v2.Start = 0.0

    # ---------------------------------------------------------------- solve
    def solve(self, time_limit):
        m = self.m
        m.Params.TimeLimit = max(0.5, float(time_limit))
        m.optimize()
        res = {"status": m.Status, "runtime": m.Runtime, "solcount": m.SolCount,
               "obj": None, "bound": None, "paths": None, "orient": None}
        if m.Status == self.GRB.INFEASIBLE:
            return res
        try:
            res["bound"] = m.ObjBound
        except self.gp.GurobiError:
            res["bound"] = None
        if m.SolCount == 0:
            return res
        res["obj"] = m.ObjVal
        inst = self.inst
        paths = {}
        for i in self.trains:
            tr = inst.trains[i]
            v, p = 0, [0]
            while tr.succ[v]:
                nxt = max(tr.succ[v], key=lambda s: self.y[(i, v, s)].X)
                p.append(nxt)
                v = nxt
            paths[i] = p
        orient = {}
        for key in self.conflicts:
            a, b = key
            vab, vba = self.zvar(a, b), self.zvar(b, a)
            xab = vab.X if vab is not None else 0.0
            xba = vba.X if vba is not None else 0.0
            if xab > 0.5:
                orient[key] = (a, b)
            elif xba > 0.5:
                orient[key] = (b, a)
        res["paths"], res["orient"] = paths, orient
        return res

    def size(self):
        self.m.update()
        return self.m.NumVars, self.m.NumConstrs
