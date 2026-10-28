import json
import sys
import gurobipy as gp
from gurobipy import GRB


INF = float('inf')

def save_solution(path, m, tr_ops, tr_op_sucs, x, y, t):
        sol = {
            "status": m.Status,
            "objective": m.ObjVal,
            "x": [[tr, op, round(t[(tr, op)].X)]
                for (tr, op) in tr_ops if x[(tr, op)].X > 0.5],
            "y": [[tr, a, b]
                for (tr, a, b) in tr_op_sucs if y[(tr, a, b)].X > 0.5],
        }
        with open(path, "w") as f:
            json.dump(sol, f)

def save_displib_solution(path, m, trains, y, t, u):
    events = []
    for tr in range(len(trains)):
        op = 0
        while True:
            events.append((round(t[(tr, op)].X), round(u[(tr, op)].X), tr, op))
            nxt = [suc for suc in trains[tr][op]['sucs'] if y[(tr, op, suc)].X > 0.5]
            if not nxt:
                break
            op = nxt[0]
    events.sort()
    sol = {"objective_value": round(m.ObjVal),
           "events": [{"time": tm, "train": tr, "operation": op} for (tm, _, tr, op) in events]}
    with open(path, "w") as f:
        json.dump(sol, f)

def get_all_edges(tr_op, trains):
    tr_op_suc = []
    for tr_op_val in tr_op:
        successors = trains[tr_op_val[0]][tr_op_val[1]]['sucs']
        for suc in successors:
            tr_op_suc.append((tr_op_val[0], tr_op_val[1], suc))
    return tr_op_suc

def get_prev_ops(tr_ops, tr_op_sucs):
    prev_ops = {}
    for (tr, op) in tr_ops:
        prev_ops[(tr, op)] = []
    for (tr, op, suc) in tr_op_sucs:
        prev_ops[(tr, suc)].append(op)
    return prev_ops

# [AI] C: horizon = latest start_lb + all durations + all release times (+1): enough time to run
# every train one after another, so an optimal schedule fits inside it. All t get UB = horizon.
# M must cover "latest possible time + release (or duration) - earliest possible time (>= 0)",
# so M = horizon + max release + max duration.
def get_big_m(trains):
    max_lb, max_release_time, max_duration = 0, 0, 0
    duration_sum, release_sum = 0, 0
    for ops in trains:
        for op in ops:
            max_lb = max(op['lb'], max_lb)
            duration_sum += op['min_duration']
            max_duration = max(max_duration, op['min_duration'])
            for res_time in op['res'].values():
                release_sum += res_time
                max_release_time = max(max_release_time, res_time)
    horizon = max_lb + duration_sum + release_sum + 1
    return horizon, horizon + max_release_time + max_duration


# [AI] A: strict=False -> pairs that overlap OR touch (need an order, used to add constraints)
#         strict=True  -> pairs that really overlap (a pair already in the model must never do this)
def detect_conflicts(tr_op_sucs, y, trains, t, strict=False):
    res_set = {}
    for (tr, op1, op2) in tr_op_sucs:
        if y[(tr, op1, op2)].X >= 0.5:
            for res_id, res_val in trains[tr][op1]['res'].items():
                start_t = round(t[(tr, op1)].X)
                end_t = round(t[(tr, op2)].X) + res_val
                res_set.setdefault(res_id, []).append((tr, op1, start_t, end_t))
    z_container = set()
    for res_list in res_set.values():
        for (tr1, op1, st1, ed1) in res_list:
            for (tr2, op2, st2, ed2) in res_list:
                if strict:
                    hit = st1 < ed2 and st2 < ed1
                else:
                    hit = st1 <= ed2 and st2 <= ed1
                if tr1 < tr2 and hit:
                    z_container.add((tr1, op1, tr2, op2))
    return sorted(z_container)

def add_resource_constraints(conflicts, trains, m, t, x, z, u, M_u, M, tr_op_sucs, y):
    for (tr1, op1, tr2, op2) in conflicts:
        z[(tr1, op1, tr2, op2)] = m.addVar(vtype=GRB.BINARY)
        z[(tr2, op2, tr1, op1)] = m.addVar(vtype=GRB.BINARY)
        ops1 = trains[tr1][op1]
        ops2 = trains[tr2][op2]
        shared_res = ops1['res'].keys() & ops2['res'].keys()
        # [AI] B: the extra term "+ M * (1 - y[edge])" switches a constraint off when the train
        # does not leave op1 towards this successor (an unused branch must not push any time)
        for res_id in shared_res:
            for op_suc1 in ops1['sucs']:
                m.addConstr(t[(tr1, op_suc1)] + ops1['res'][res_id] <= t[(tr2, op2)] + M * (1 - z[(tr1, op1, tr2, op2)]) + M * (1 - y[(tr1, op1, op_suc1)]))
            for op_suc2 in ops2['sucs']:
                m.addConstr(t[(tr2, op_suc2)] + ops2['res'][res_id] <= t[(tr1, op1)] + M * (1 - z[(tr2, op2, tr1, op1)]) + M * (1 - y[(tr2, op2, op_suc2)]))

        for op_suc1 in ops1['sucs']:
            m.addConstr(u[(tr1, op_suc1)] + 1 <= u[(tr2, op2)] + M_u * (1 - z[(tr1, op1, tr2, op2)]) + M_u * (1 - y[(tr1, op1, op_suc1)]))
        for op_suc2 in ops2['sucs']:
            m.addConstr(u[(tr2, op_suc2)] + 1 <= u[(tr1, op1)] + M_u * (1 - z[(tr2, op2, tr1, op1)]) + M_u * (1 - y[(tr2, op2, op_suc2)]))
        m.addConstr(x[(tr1, op1)] >= z[(tr1, op1, tr2, op2)] + z[(tr2, op2, tr1, op1)]) # Zab+Zba is at most 1
        m.addConstr(x[(tr2, op2)] >= z[(tr1, op1, tr2, op2)] + z[(tr2, op2, tr1, op1)])
        m.addConstr(x[(tr1, op1)] + x[(tr2, op2)] - 1 <= z[(tr1, op1, tr2, op2)] + z[(tr2, op2, tr1, op1)]) # if both x are 1, it forces one of the z be 1


def parse(path):
    prob = path
    with open(prob) as f:
        raw = json.load(f)
    trains = []
    last_tr_ops = []
    tr_ops = []
    for tr_ind, tr in enumerate(raw['trains']):
        ops = []
        for op_ind, op in enumerate(tr):
            ops.append({
                'lb': op.get('start_lb', 0),
                'ub': op.get('start_ub', INF),
                'min_duration': op.get('min_duration', 0),
                'res': {r['resource']: r.get('release_time', 0) for r in op.get('resources', [])},
                'sucs': op.get('successors', [])
            })
            tr_ops.append((tr_ind, op_ind))
            if len(ops[-1]['sucs']) == 0:
                last_tr_ops.append((tr_ind, op_ind))
        trains.append(ops)
    objectives = raw['objective']
    objs = []
    for obj in objectives:
        objs.append({
            'tr': obj['train'],
            'op': obj['operation'],
            'threshold': obj.get('threshold', 0),
            'coeff': obj.get('coeff', 0)
        })
    return trains, objs, tr_ops, last_tr_ops

def solve(path, out_path="solution_displib.json"):
    trains, objs, tr_ops, last_tr_ops = parse(path)
    tr_op_sucs = get_all_edges(tr_ops, trains)
    prev_ops = get_prev_ops(tr_ops, tr_op_sucs)
    m = gp.Model('DISPLIB_MILP')
    m.Params.OutputFlag = 0
    H, M = get_big_m(trains)  # [AI] C: horizon H (upper bound of every t) and big-M
    M_u = len(tr_ops)
    t = m.addVars(tr_ops, vtype=GRB.INTEGER, name="time")
    x = m.addVars(tr_ops, vtype=GRB.BINARY, name="operation")
    y = m.addVars(tr_op_sucs, vtype=GRB.BINARY, name="edge")
    w = m.addVars(len(objs), lb = 0, vtype=GRB.CONTINUOUS, name="penalty")
    z = {}
    u = m.addVars(tr_ops, vtype=GRB.INTEGER, name="order")

    # first operation (start)
    for tr_ind in range(len(trains)):
        m.addConstr(gp.quicksum(y[(tr_ind, 0, suc)] for suc in trains[tr_ind][0]['sucs']) == 1)

    # last operation constraint (end)
    for (last_tr, last_op) in last_tr_ops:
        m.addConstr(gp.quicksum(y[(last_tr, pred, last_op)] for pred in prev_ops[(last_tr, last_op)]) == 1) 

    # routing constraint
    for (tr, op, suc) in tr_op_sucs:
        yab = y[(tr, op, suc)]
        xa = x[(tr,op)]
        xb = x[(tr, suc)]
        m.addConstr(yab <= xa)
        m.addConstr(yab <= xb)
        m.addConstr(xa+xb-1<=yab)
        # [AI] B: was  t[suc] >= t[op] + dur * y  -> with y = 0 it still forced t[suc] >= t[op],
        # so an unused op (with its start_lb) pushed the op where the branches join
        m.addConstr(t[(tr, suc)] >= t[(tr, op)] + trains[tr][op]['min_duration'] - M * (1 - yab))
        m.addConstr(u[(tr, op)] + 1 <= u[(tr, suc)] + M_u * (1 - y[(tr, op, suc)])) # if successive operation is next it has to be have order number more than previous

    # objective
    for k in range(len(objs)):
        tr = objs[k]['tr']
        op = objs[k]['op']
        m.addConstr(w[k] >= objs[k]['coeff'] * (t[(tr, op)] - objs[k]['threshold']))

    # flow constraint (in-out) + time_constraint (start_lb <= time <= start_ub)
    for (tr, op) in tr_ops:
        t[(tr, op)].LB = trains[tr][op]['lb']
        t[(tr, op)].UB = min(trains[tr][op]['ub'], H)  # [AI] C: every t inside the horizon, so M is valid
        if len(trains[tr][op]['sucs']) == 0 or len(prev_ops[(tr, op)]) == 0:
            continue
        prev = prev_ops[(tr, op)]
        post = trains[tr][op]['sucs']
        # prev - list of previous operations, post - list of successor operations
        m.addConstr(gp.quicksum(y[tr, prev_op, op] for prev_op in prev) == gp.quicksum(y[tr, op, post_op] for post_op in post)) 

    # resource constraints
    

    m.setObjective(gp.quicksum(w[i] for i in range(len(objs))), GRB.MINIMIZE)


    m.Params.MIPGap = 0         # [AI] C: was 0.05 (5 %); 0 = prove the real optimum
    m.Params.TimeLimit = 300    # only 300 seconds for run
    m.Params.IntFeasTol = 1e-9  # [AI] C: default 1e-5 accepts z = 0.99999 as 1 -> M * 1e-5 seconds of hidden slack
    added = set()
    # [AI] A: a pair already in `added` may come back (touching times are allowed) - that is normal.
    # Only NEW pairs are added; when there are none, the solution is conflict-free.
    while True:
        m.optimize()
        if m.Status == GRB.INFEASIBLE or m.SolCount == 0:
            print('🚨 ERROR: The model is mathematically impossible (Infeasible). Check your constraints!')
            break
        conflicts = detect_conflicts(tr_op_sucs, y, trains, t)
        new_conflicts = [c for c in conflicts if c not in added]
        print(f'objective {m.ObjVal:.0f}: {len(conflicts)} pairs need an order, {len(new_conflicts)} of them new')
        # a pair that is already in the model but really overlaps = model error (not just touching)
        broken = [c for c in detect_conflicts(tr_op_sucs, y, trains, t, strict=True) if c in added]
        if broken:
            print(f'WARNING: {len(broken)} pairs already in the model still overlap, e.g. {broken[0]}')
        save_solution("solution.json", m, tr_ops, tr_op_sucs, x, y, t)
        save_displib_solution(out_path, m, trains, y, t, u)
        if len(new_conflicts) == 0:
            if m.Status == GRB.OPTIMAL:
                print('found optimal solution')
            else:
                print('found feasible solution (time limit: optimality not proven)')
            break
        add_resource_constraints(new_conflicts, trains, m, t, x, z, u, M_u, M, tr_op_sucs, y)
        added.update(new_conflicts)
        



if __name__ == '__main__':
    # [AI] D: optional 2nd argument = output file in DISPLIB format (default solution_displib.json)
    solve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "solution_displib.json")