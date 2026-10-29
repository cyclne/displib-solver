"""
A tiny gurobipy-like modelling layer on top of HiGHS (highspy).

AI-generated (Claude), October 2026.

Purpose: run and test the solver without a Gurobi licence (e.g. Gurobi's free pip licence
is limited to 2000 variables/constraints). It implements only the subset of the gurobipy
API that model.py uses. Gurobi remains the main backend; HiGHS is much slower on hard
instances but gives the same optimal values.
"""

import time

import numpy as np

try:
    import highspy
except ImportError:  # pragma: no cover
    highspy = None


class GRB:
    BINARY = "B"
    CONTINUOUS = "C"
    MINIMIZE = 1
    OPTIMAL = 2
    INFEASIBLE = 3
    TIME_LIMIT = 9
    OTHER = 99


class GurobiError(Exception):
    pass


class LinExpr:
    __slots__ = ("coef", "const")

    def __init__(self, const=0.0):
        self.coef = {}
        self.const = float(const)

    def copy(self):
        e = LinExpr(self.const)
        e.coef = dict(self.coef)
        return e

    def _add(self, other, sign):
        if isinstance(other, LinExpr):
            for k, v in other.coef.items():
                self.coef[k] = self.coef.get(k, 0.0) + sign * v
            self.const += sign * other.const
        elif isinstance(other, Var):
            self.coef[other.idx] = self.coef.get(other.idx, 0.0) + sign
        else:
            self.const += sign * float(other)
        return self

    def __iadd__(self, o):
        return self._add(o, 1.0)

    def __isub__(self, o):
        return self._add(o, -1.0)

    def __add__(self, o):
        return self.copy()._add(o, 1.0)

    __radd__ = __add__

    def __sub__(self, o):
        return self.copy()._add(o, -1.0)

    def __rsub__(self, o):
        return (-self)._add(o, 1.0)

    def __neg__(self):
        e = LinExpr(-self.const)
        e.coef = {k: -v for k, v in self.coef.items()}
        return e

    def __mul__(self, c):
        c = float(c)
        e = LinExpr(self.const * c)
        e.coef = {k: v * c for k, v in self.coef.items()}
        return e

    __rmul__ = __mul__

    def __le__(self, o):
        return _Temp(self - o, "<")

    def __ge__(self, o):
        return _Temp(self - o, ">")

    def __eq__(self, o):
        return _Temp(self - o, "=")

    __hash__ = None

    def getValue(self):
        raise GurobiError("getValue needs the model; use Model.value(expr)")


class Var:
    __slots__ = ("model", "idx")

    def __init__(self, model, idx):
        self.model = model
        self.idx = idx

    def _e(self):
        e = LinExpr()
        e.coef[self.idx] = 1.0
        return e

    def __add__(self, o):
        return self._e() + o

    __radd__ = __add__

    def __sub__(self, o):
        return self._e() - o

    def __rsub__(self, o):
        return (-self._e()) + o

    def __neg__(self):
        return -self._e()

    def __mul__(self, c):
        return self._e() * c

    __rmul__ = __mul__

    def __le__(self, o):
        return self._e() <= o

    def __ge__(self, o):
        return self._e() >= o

    def __eq__(self, o):
        return self._e() == o

    __hash__ = object.__hash__

    @property
    def X(self):
        return self.model._x[self.idx]

    @property
    def VType(self):
        return self.model._vt[self.idx]

    @property
    def LB(self):
        return self.model._lb[self.idx]

    @LB.setter
    def LB(self, v):
        self.model._lb[self.idx] = float(v)

    @property
    def UB(self):
        return self.model._ub[self.idx]

    @UB.setter
    def UB(self, v):
        self.model._ub[self.idx] = float(v)

    @property
    def Start(self):
        return self.model._start.get(self.idx)

    @Start.setter
    def Start(self, v):
        self.model._start[self.idx] = float(v)


class _Temp:
    def __init__(self, expr, sense):
        self.expr, self.sense = expr, sense


def quicksum(it):
    e = LinExpr()
    for t in it:
        e += t
    return e


class _Params:
    def __init__(self):
        self.TimeLimit = 1e30
        self.MIPGap = 1e-4
        self.IntFeasTol = 1e-6
        self.Threads = 0
        self.OutputFlag = 0


class Model:
    def __init__(self, name="", env=None):
        if highspy is None:
            raise GurobiError("highspy is not installed (pip install highspy)")
        self.Params = _Params()
        self._lb, self._ub, self._vt, self._names = [], [], [], []
        self._rows = []           # (coef dict, lo, hi)
        self._obj = LinExpr()
        self._start = {}
        self._x = []
        self.Status, self.SolCount, self.ObjVal, self.ObjBound, self.Runtime = None, 0, None, None, 0.0

    def addVar(self, lb=0.0, ub=float("inf"), vtype="C", name=""):
        idx = len(self._lb)
        if vtype == "B":
            lb, ub = max(0.0, lb), min(1.0, ub)
        self._lb.append(float(lb))
        self._ub.append(float(ub))
        self._vt.append(vtype)
        self._names.append(name)
        return Var(self, idx)

    def addConstr(self, tc, name=""):
        e = tc.expr
        rhs = -e.const
        if tc.sense == "<":
            lo, hi = -np.inf, rhs
        elif tc.sense == ">":
            lo, hi = rhs, np.inf
        else:
            lo, hi = rhs, rhs
        self._rows.append((dict(e.coef), lo, hi))
        return len(self._rows) - 1

    def setObjective(self, expr, sense=GRB.MINIMIZE):
        self._obj = expr if isinstance(expr, LinExpr) else quicksum([expr])

    def update(self):
        pass

    @property
    def NumVars(self):
        return len(self._lb)

    @property
    def NumConstrs(self):
        return len(self._rows)

    def value(self, expr):
        if isinstance(expr, Var):
            return expr.X
        return expr.const + sum(v * self._x[k] for k, v in expr.coef.items())

    def optimize(self):
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(self.Params.OutputFlag))
        h.setOptionValue("time_limit", float(min(self.Params.TimeLimit, 1e30)))
        h.setOptionValue("mip_rel_gap", float(self.Params.MIPGap))
        h.setOptionValue("mip_feasibility_tolerance", max(1e-9, float(self.Params.IntFeasTol)))
        if self.Params.Threads:
            h.setOptionValue("threads", int(self.Params.Threads))
        n = len(self._lb)
        cost = np.zeros(n)
        for k, v in self._obj.coef.items():
            cost[k] += v
        lb = np.array(self._lb, dtype=float)
        ub = np.array(self._ub, dtype=float)
        lb[np.isinf(lb)] = -highspy.kHighsInf
        ub[np.isinf(ub)] = highspy.kHighsInf
        h.addCols(n, cost, lb, ub, 0, np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                  np.array([], dtype=float))
        h.changeObjectiveOffset(self._obj.const)
        ints = [k for k, t in enumerate(self._vt) if t == "B"]
        if ints:
            h.changeColsIntegrality(len(ints), np.array(ints, dtype=np.int32),
                                    np.array([highspy.HighsVarType.kInteger] * len(ints)))
        m = len(self._rows)
        if m:
            starts, idx, val, lo, hi = [], [], [], [], []
            for (coef, l, u) in self._rows:
                starts.append(len(idx))
                for k, v in coef.items():
                    if v != 0.0:
                        idx.append(k)
                        val.append(v)
                lo.append(-highspy.kHighsInf if l == -np.inf else l)
                hi.append(highspy.kHighsInf if u == np.inf else u)
            h.addRows(m, np.array(lo, dtype=float), np.array(hi, dtype=float), len(idx),
                      np.array(starts, dtype=np.int32), np.array(idx, dtype=np.int32),
                      np.array(val, dtype=float))
        if self._start:
            ks = sorted(self._start)
            try:
                h.setSolution(len(ks), np.array(ks, dtype=np.int32),
                              np.array([self._start[k] for k in ks], dtype=float))
            except Exception:
                pass
        t0 = time.time()
        h.run()
        self.Runtime = time.time() - t0
        st = h.getModelStatus()
        info = h.getInfo()
        MS = highspy.HighsModelStatus
        has_sol = info.primal_solution_status == 2
        if st == MS.kOptimal:
            self.Status = GRB.OPTIMAL
        elif st == MS.kInfeasible:
            self.Status = GRB.INFEASIBLE
        elif st == MS.kTimeLimit:
            self.Status = GRB.TIME_LIMIT
        else:
            self.Status = GRB.OTHER
        self.SolCount = 1 if has_sol else 0
        if has_sol:
            self._x = list(h.getSolution().col_value)
            self.ObjVal = info.objective_function_value
        else:
            self._x = []
            self.ObjVal = None
        bound = info.mip_dual_bound if ints else info.objective_function_value
        self.ObjBound = bound if bound is not None and abs(bound) < 1e29 else None
        if self.Status == GRB.OPTIMAL and self.ObjBound is None:
            self.ObjBound = self.ObjVal
