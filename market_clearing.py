# market_clearing.py

import pyomo.environ as pyo

def build_model(config, allow_simultaneous_import_export=False, allow_cross_flows=False):
    """
    Build a Pyomo model from:
      config = {
        'nodes':   [...],
        'line_cap': {(i,j):cap, ...},    # capacities for each undirected edge
        'demand':  {i:Di, ...},
        'bids':    {i: [(q,p), ...], ...}
      }
    Returns a ConcreteModel with:
      m.N  = node set
      m.U  = undirected edge set (each tuple sorted)
      m.S  = bid‐segment index set
      m.x  = dispatch vars on m.S
      m.f  = signed flow vars on m.U
      constraints: seg_limit, flow_cap, balance, [no_export, no_import]
      m.obj = minimize total procurement cost
    """
    nodes    = config['nodes']
    line_cap = config['line_cap']
    D        = config['demand']
    bids     = config['bids']

    # 1) count bid segments per node
    seg_count = {i: len(bids[i]) for i in nodes}
    # 2) build undirected edge list (sort each tuple)
    U = list({tuple(sorted(e)) for e in line_cap})

    m = pyo.ConcreteModel()
    m.N = pyo.Set(initialize=nodes)
    m.U = pyo.Set(initialize=U, dimen=2)
    m.S = pyo.Set(
        initialize=[(i, k) for i in nodes for k in range(seg_count[i])],
        dimen=2
    )

    # Parameters
    m.q = pyo.Param(m.S, initialize=lambda m, i, k: bids[i][k][0])
    m.p = pyo.Param(m.S, initialize=lambda m, i, k: bids[i][k][1])
    m.D = pyo.Param(m.N, initialize=D)

    def cap_init(m, i, j):
        return line_cap.get((i, j), line_cap.get((j, i)))
    m.Fbar = pyo.Param(m.U, initialize=cap_init)

    # Variables
    m.x = pyo.Var(m.S, within=pyo.NonNegativeReals)
    m.f = pyo.Var(m.U, within=pyo.Reals)

    # 1) bid‐segment limits
    m.seg_limit = pyo.Constraint(
        m.S,
        rule=lambda m, i, k: m.x[i, k] <= m.q[i, k]
    )

    # 2) line capacity: -Fbar <= f <= Fbar
    m.flow_cap = pyo.Constraint(
        m.U,
        rule=lambda m, i, j: (-m.Fbar[i, j], m.f[i, j], m.Fbar[i, j])
    )

    # 3) energy balance: generation + imports = demand + exports
    def balance_rule(m, i):
        gen     = sum(m.x[i, k] for (ii, k) in m.S if ii == i)
        inflow  = sum(m.f[j, i] for (j, i2) in m.U if i2 == i)
        outflow = sum(m.f[i, j] for (i2, j) in m.U if i2 == i)
        return gen + inflow == m.D[i] + outflow
    m.balance = pyo.Constraint(m.N, rule=balance_rule)

    # Objective: minimize total procurement cost
    m.obj = pyo.Objective(
        expr=sum(m.p[i, k] * m.x[i, k] for (i, k) in m.S),
        sense=pyo.minimize
    )

    return m
    
def solve_model(m):
    """
    Solve the model and return a results dict:
      {
        'status': str,
        'cost': float,
        'supply': {node: supply},
        'import': {node: import},
        'export': {node: export},
        'flows': {(i,j): flow},
        'bids': {(node, segment): accepted_quantity}
      }
    """
    res = pyo.SolverFactory('appsi_highs').solve(m, tee=False)
    status = str(res.solver.termination_condition)
    cost   = pyo.value(m.obj)

    supply = {}
    imp    = {}
    exp    = {}
    for node in m.N:
        # total supply at node
        supply[node] = sum(pyo.value(m.x[i,k]) for i,k in m.S if i == node)
        # import (positive incoming flows)
        imp[node]    = sum(pyo.value(m.f[j,node]) for j in m.N
                             if (j,node) in m.Fbar and pyo.value(m.f[j,node]) > 0)
        # export (positive outgoing flows)
        exp[node]    = sum(pyo.value(m.f[node,j]) for j in m.N
                             if (node,j) in m.Fbar and pyo.value(m.f[node,j]) > 0)

    # record individual bid segment acceptances
    bids = {}
    for i,k in m.S:
        bids[(i, k)] = pyo.value(m.x[i, k])

    flows = {}
    for i, j in m.U:
        flows[(i, j)] = pyo.value(m.f[i, j])

    return {
        'status': status,
        'cost': cost,
        'supply': supply,
        'import': imp,
        'export': exp,
        'flows': flows,
        'bids': bids,
    }
