"""
Cuts on the orientation variables z (paper Section 5.3).

AI-generated (Claude), October 2026.

1) Order-agreement cuts (21):  z[rhoA, rhoB] + z[sigmaB, sigmaA] <= 1
   for two conflicts {rhoA, rhoB} and {sigmaA, sigmaB} between the same trains A and B.
   The paper derives them from "overlapping / forward- / backward-implying" resource
   usages but does not spell out the test. Here they are added only when one of two
   sufficient conditions holds (each proven in README.md):

   Lemma A (same direction):  hold_A(rho, sigma)  and  before_B(rho, sigma)
   Lemma B (opposite):        hold_A(rho, sigma)  and  hold_B(sigma, rho)

   The proofs use the DISPLIB event sequence (a strict total order of start events; an
   operation's end event is the start event of the train's next operation), so they also
   hold for zero durations and zero release times.

   hold_T(rho, sigma):   T's reservation event of sigma is no later than T's last end event
                         of rho: every reserving op e of sigma lies in rho's run, or all
                         predecessors of e lie in rho's run (e is the op right after the run).
   before_T(rho, sigma): on every path, T reserves rho no later than sigma
                         (no op of sigma outside rho can reach an op of rho).

   "X before Y on rho" (z = 1) means: X's end events of rho precede Y's start event of rho.
   Proof of A: start_A(sigma) <= end_A(rho) < start_B(rho) <= start_B(sigma) <= end_B(sigma);
   so end_B(sigma) < start_A(sigma) (B before A on sigma) would close a cycle.
   Proof of B: start_A(sigma) <= end_A(rho) < start_B(rho) <= end_B(sigma) < start_A(sigma)
   if B were before A on sigma: a cycle again.

2) No-swap cuts: for a zero-cost cycle observed in a solution (a swap), forbid the
   combination of arcs that created it:  sum of the binaries on the cycle <= count - 1.
   This is the "add only the cycles that appear" variant mentioned in the paper.
"""


class CutFactory:
    def __init__(self, inst, route_conditional=True):
        self.inst = inst
        self.route_conditional = route_conditional
        self._hold = {}
        self._before = {}

    # ------------------------------------------------------------ structural tests
    def hold(self, rho, sigma):
        key = (rho, sigma)
        if key in self._hold:
            return self._hold[key]
        inst = self.inst
        ur, us = inst.usages[rho], inst.usages[sigma]
        ok = False
        if rho != sigma and ur.train == us.train and us.rin:
            tr = inst.trains[ur.train]
            ok = True
            for e in us.rin:
                # sigma is reserved inside rho's run, or by the op right after rho's run
                # (that start event is also rho's last end event)
                if e in ur.ops:
                    continue
                if e != 0 and tr.pred[e] and all(p in ur.ops for p in tr.pred[e]):
                    continue
                ok = False
                break
        self._hold[key] = ok
        return ok

    def before(self, rho, sigma):
        key = (rho, sigma)
        if key in self._before:
            return self._before[key]
        inst = self.inst
        ur, us = inst.usages[rho], inst.usages[sigma]
        ok = False
        if rho != sigma and ur.train == us.train:
            tr = inst.trains[ur.train]
            mask = 0
            for v in ur.ops:
                mask |= 1 << v
            R = 0
            for v in us.ops:
                if v not in ur.ops:
                    R |= tr.reach[v]
            ok = (R & mask) == 0
        self._before[key] = ok
        return ok

    # ------------------------------------------------------------ agreement cuts (21)
    def agreement_cuts(self, k1, k2):
        """Cuts between conflict keys k1 and k2 (each a (uid, uid) pair of two trains)."""
        inst = self.inst
        by1 = {inst.usages[u].train: u for u in k1}
        by2 = {inst.usages[u].train: u for u in k2}
        if set(by1) != set(by2) or len(by1) != 2:
            return []
        i, j = sorted(by1)
        cuts = []
        for (A, B) in ((i, j), (j, i)):
            for (P, Q) in ((by1, by2), (by2, by1)):
                rA, rB, sA, sB = P[A], P[B], Q[A], Q[B]
                if rA == sA or rB == sB:
                    continue
                if self.hold(rA, sA) and (self.before(rB, sB) or self.hold(sB, rB)):
                    cid = ("agree", rA, rB, sB, sA)
                    cuts.append((cid, [("z", rA, rB), ("z", sB, sA)], 1, {A, B}))
                elif self.route_conditional:
                    cuts += self._route_cuts(A, B, rA, rB, sA, sB)
        return cuts

    def _route_cuts(self, A, B, rA, rB, sA, sB, limit=4):
        """
        Route-conditional version of the same two lemmas, for when the structural tests fail
        because some *other* route breaks them:
          A uses an arc (p, e) leaving rho's run into sigma  (p in rhoA, e not in rhoA, e in sigmaA)
          B uses an arc (q, f) that either
            - leaves sigma's run into rho  (q in sigmaB, f not in sigmaB, f in rhoB)   [Lemma B], or
            - enters sigma directly from rho (q in rhoB, q not in sigmaB, f in sigmaB) [Lemma A].
        On these arcs the event chain of the lemma closes, so
            z[rhoA, rhoB] + z[sigmaB, sigmaA] + y_A[p, e] + y_B[q, f] <= 3.
        These are exactly the two-train swap cycles that would otherwise be found reactively.
        """
        inst = self.inst
        uA_r, uA_s = inst.usages[rA], inst.usages[sA]
        uB_r, uB_s = inst.usages[rB], inst.usages[sB]
        trA, trB = inst.trains[A], inst.trains[B]
        a_arcs = [(p, e) for p in uA_r.ops for e in trA.succ[p]
                  if e not in uA_r.ops and e in uA_s.ops]
        if not a_arcs:
            return []
        b_arcs = [(q, f) for q in uB_s.ops for f in trB.succ[q]
                  if f not in uB_s.ops and f in uB_r.ops]
        b_arcs += [(q, f) for q in uB_r.ops if q not in uB_s.ops for f in trB.succ[q]
                   if f in uB_s.ops]
        out = []
        for (p, e) in a_arcs:
            for (q, f) in b_arcs:
                cid = ("agree_r", rA, rB, sB, sA, p, e, q, f)
                out.append((cid, [("z", rA, rB), ("z", sB, sA), ("y", A, p, e), ("y", B, q, f)], 3, {A, B}))
                if len(out) >= limit:
                    return out
        return out

    # ------------------------------------------------------------ no-swap cut
    def noswap_cut(self, cycle, paths):
        """
        cycle: list of (u, v, info) arcs from timing.order_events.
        Returns (cid, terms, rhs, trains, conflict_keys_needed).
        """
        inst = self.inst
        terms, need, trains = set(), set(), set()
        for (_, _, info) in cycle:
            if info[0] == "train":
                _, i, a, b = info
                terms.add(("y", i, a, b))
                trains.add(i)
            else:
                _, ua, ub, i, k, nxt = info
                terms.add(("z", ua, ub))
                terms.add(("y", i, k, nxt))
                j = inst.usages[ub].train
                p = paths[j]
                first = min((v for v in inst.usages[ub].ops if v in p), key=p.index)
                idx = p.index(first)
                if idx > 0:
                    terms.add(("y", j, p[idx - 1], first))
                need.add((min(ua, ub), max(ua, ub)))
                trains.add(i)
                trains.add(j)
        terms = sorted(terms)
        cid = ("swap",) + tuple(terms)
        return cid, terms, len(terms) - 1, trains, need
