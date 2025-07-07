# market_clearing.py

import pyomo.environ as pyo

def build_model(config, allow_simultaneous_import_export=False):
    """
    Pyomo model with per-node importer/exporter logic via two nonnegative
    directed flows per undirected line (f_plus and f_minus).
    """

    # Unpack data
    nodes    = config['nodes']      # list of node IDs
    line_cap = config['line_cap']   # dict { (i,j) or (j,i) : capacity }
    D        = config['demand']     # dict { i: demand }
    bids     = config['bids']       # dict { i: [ (q,p), ... ] }

    # Build the set of sorted undirected edges
    U = list({tuple(sorted(e)) for e in line_cap})

    # Count bid-segments per node
    seg_count = {i: len(bids[i]) for i in nodes}

    m = pyo.ConcreteModel()
    m.N = pyo.Set(initialize=nodes)
    m.U = pyo.Set(initialize=U, dimen=2)               # undirected edges (i<j)
    m.S = pyo.Set(                                      # bid segments
        initialize=[(i,k) for i in nodes for k in range(seg_count[i])],
        dimen=2
    )

    # Parameters
    m.q = pyo.Param(m.S, initialize=lambda m,i,k: bids[i][k][0])
    m.p = pyo.Param(m.S, initialize=lambda m,i,k: bids[i][k][1])
    m.D = pyo.Param(m.N, initialize=D)
    def cap_init(m,i,j):
        # look up undirected capacity, whether (i,j) or (j,i)
        return line_cap.get((i,j), line_cap.get((j,i), 0))
    m.Fbar = pyo.Param(m.U, initialize=cap_init)

    # Variables
    m.x        = pyo.Var(m.S, within=pyo.NonNegativeReals)  # generation
    m.f_plus   = pyo.Var(m.U, within=pyo.NonNegativeReals)  # flow i→j
    m.f_minus  = pyo.Var(m.U, within=pyo.NonNegativeReals)  # flow j→i

    # 1) Bid segment limits
    m.seg_limit = pyo.Constraint(
        m.S, rule=lambda m,i,k: m.x[i,k] <= m.q[i,k]
    )

    # 2) Per-direction capacity
    m.cap_plus  = pyo.Constraint(
        m.U, rule=lambda m,i,j: m.f_plus[i,j]  <= m.Fbar[i,j]
    )
    m.cap_minus = pyo.Constraint(
        m.U, rule=lambda m,i,j: m.f_minus[i,j] <= m.Fbar[i,j]
    )

    # 3) Node balance: gen + imports = demand + exports
    def balance_rule(m,node):
        gen = sum(m.x[node,k] for (ii,k) in m.S if ii==node)

        # imports into node:
        #   from j>node   ⇒ f_minus[node,j]
        #   from j<node   ⇒ f_plus[j,node]
        inflow = sum(m.f_minus[node,j]    for (i,j) in m.U if i   == node) \
               + sum(m.f_plus[j,node]     for (j,i) in m.U if i   == node)

        # exports out of node:
        #   to j>node     ⇒ f_plus[node,j]
        #   to j<node     ⇒ f_minus[j,node]
        outflow= sum(m.f_plus[node,j]     for (i,j) in m.U if i   == node) \
               + sum(m.f_minus[j,node]    for (j,i) in m.U if i   == node)

        return gen + inflow == m.D[node] + outflow

    m.balance = pyo.Constraint(m.N, rule=balance_rule)

    # 4–5) Forbid simultaneous import & export
    if not allow_simultaneous_import_export:
        # 1) importer‐flag
        m.is_importer = pyo.Var(m.N, within=pyo.Binary)

        # 2) big‐M per node
        def M_init(m, i):
            return sum(m.Fbar[i,j] for (i2,j) in m.U if i2==i) + \
                   sum(m.Fbar[j,i] for (j,i2) in m.U if i2==i)
        m.M = pyo.Param(m.N, initialize=M_init)

        # 3) forbid imports when is_importer=0
        def forbid_imports(m, i):
            imports = sum(m.f_minus[i,j] for (i2,j) in m.U if i2==i) + \
                      sum(m.f_plus[j,i]  for (j,i2) in m.U if i2==i)
            return imports <= m.M[i] * m.is_importer[i]
        m.forbid_imports = pyo.Constraint(m.N, rule=forbid_imports)

        # 4) forbid exports when is_importer=1
        def forbid_exports(m, i):
            exports = sum(m.f_plus[i,j]   for (i2,j) in m.U if i2==i) + \
                      sum(m.f_minus[j,i]  for (j,i2) in m.U if i2==i)
            return exports <= m.M[i] * (1 - m.is_importer[i])
        m.forbid_exports = pyo.Constraint(m.N, rule=forbid_exports)

    # 6) Objective: minimize generation cost
    m.obj = pyo.Objective(
        expr=sum(m.p[i,k] * m.x[i,k] for (i,k) in m.S),
        sense=pyo.minimize
    )

    return m
    
def solve_all_cases(config, solver_name='appsi_highs'):
    """
    Solves the model in three stages:
    
    1) Baseline solve allowing simultaneous import/export (m0) to get reference values.
    2) Solve with no simultaneous import/export (m1).
       Extract key data: generation, import quantities, importer flags.
    3) Solve with simultaneous import/export (m2), adding delta constraints
       only for prior-importer nodes and linking to prior model results.

    Returns:
        results0: Solver results for baseline case m0
        m0: Pyomo model object for baseline
        results1: Solver results for m1
        m1: Pyomo model object for m1
        results2: Solver results for m2
        m2: Pyomo model object for m2
    """

    solver = pyo.SolverFactory(solver_name)

    # ─────────────────────────────────────────────────────────────
    # ─── Stage 0: Baseline run with simultaneous import/export ───
    # ─────────────────────────────────────────────────────────────
    m0 = build_model(config, allow_simultaneous_import_export=True)
    results0 = solver.solve(m0)

    # ─────────────────────────────────────────────────────────────
    # ─── Stage 1: Disable simultaneous import/export (m1) ─────────
    # ─────────────────────────────────────────────────────────────
    m1 = build_model(config, allow_simultaneous_import_export=False)
    results1 = solver.solve(m1)

    # Extract total generation and import for each node in m1
    old_gen = {
        i: sum(m1.x[i, k].value for (ii, k) in m1.S if ii == i)
        for i in m1.N
    }
    old_imp = {
        i: sum(m1.f_minus[i, j].value for (i2, j) in m1.U if i2 == i)
           + sum(m1.f_plus[j, i].value for (j, i2) in m1.U if i2 == i)
        for i in m1.N
    }

    # Identify nodes that were net importers in m1
    importer_flag = {
        i: 1 if old_imp[i] > 1e-6 else 0
        for i in m1.N
    }

    # ─────────────────────────────────────────────────────────────
    # ─── Stage 2: Enable simultaneous import/export again (m2) ────
    # ─────────────────────────────────────────────────────────────
    m2 = build_model(config, allow_simultaneous_import_export=True)

    # Store old data from m1 as immutable model parameters
    m2.old_gen       = pyo.Param(m2.N, initialize=old_gen,       mutable=False)
    m2.old_imp       = pyo.Param(m2.N, initialize=old_imp,       mutable=False)
    m2.importer_flag = pyo.Param(m2.N, initialize=importer_flag, mutable=False)

    # Big-M parameter: max flow capacity incident on each node
    def M_init(m, i):
        return sum(m.Fbar[i, j] for (i2, j) in m.U if i2 == i) \
             + sum(m.Fbar[j, i] for (j, i2) in m.U if i2 == i)
    m2.M = pyo.Param(m2.N, initialize=M_init, mutable=False)

    # Binary variable: 1 if generation at node decreased vs m1
    m2.gen_decrease = pyo.Var(m2.N, within=pyo.Binary)

    # Helper expressions
    def new_imp(m, i):
        return sum(m.f_minus[i, j] for (i2, j) in m.U if i2 == i) \
             + sum(m.f_plus[j, i] for (j, i2) in m.U if i2 == i)

    def new_gen(m, i):
        return sum(m.x[i, k] for (ii, k) in m.S if ii == i)

    # Constraint 1: detect if generation decreased
    m2.gendec1 = pyo.Constraint(m2.N, rule=lambda m, i:
        m.old_gen[i] - new_gen(m, i) <= m.M[i] * m.gen_decrease[i])

    m2.gendec2 = pyo.Constraint(m2.N, rule=lambda m, i:
        new_gen(m, i) - m.old_gen[i] <= m.M[i] * (1 - m.gen_decrease[i]))

    # Variable: increase in imports (delta) vs m1
    m2.delta_imp = pyo.Var(m2.N, within=pyo.NonNegativeReals)

    # Constraint 2: delta must cover any increase in import
    m2.link_delta = pyo.Constraint(m2.N, rule=lambda m, i:
        m.delta_imp[i] >= new_imp(m, i) - m.old_imp[i])

    # Constraint 3a: limit delta by generation drop if gen_decrease == 1
    m2.cap1 = pyo.Constraint(m2.N, rule=lambda m, i:
        m.delta_imp[i] <= (m.old_gen[i] - new_gen(m, i)) 
                          + m.M[i] * (1 - m.gen_decrease[i]))

    # Constraint 3b: delta must be 0 if generation did not decrease
    m2.cap2 = pyo.Constraint(m2.N, rule=lambda m, i:
        m.delta_imp[i] <= m.M[i] * m.gen_decrease[i])

    # Constraint 4: only prior importers are allowed to increase imports
    m2.cap_importer = pyo.Constraint(m2.N, rule=lambda m, i:
        new_imp(m, i) <= m.old_imp[i] + m.delta_imp[i]
        if m.importer_flag[i] == 1 else pyo.Constraint.Skip)

    # Solve final model
    results2 = solver.solve(m2)

    # Pack into a dictionary with results
    return {"baseline": get_results(m0, results0),
            "no_simultaneous": get_results(m1, results1),
            "new": get_results(m2, results2)}

def get_results(model, result):
    status = str(result.solver.termination_condition)
    cost   = pyo.value(model.obj)

    # total generation per node
    supply = {
        node: sum(
            pyo.value(model.x[i, k])
            for (i, k) in model.S
            if i == node
        )
        for node in model.N
    }

    # imports/exports per node
    imp = {}
    exp = {}
    for node in model.N:
        # imports into `node`
        imports = sum(
            pyo.value(model.f_minus[node, j])
            for (i, j) in model.U
            if i == node
        ) + sum(
            pyo.value(model.f_plus[j, node])
            for (j, i) in model.U
            if i == node
        )
        imp[node] = imports

        # exports out of `node`
        exports = sum(
            pyo.value(model.f_plus[node, j])
            for (i, j) in model.U
            if i == node
        ) + sum(
            pyo.value(model.f_minus[j, node])
            for (j, i) in model.U
            if i == node
        )
        exp[node] = exports

    # bid‐segment dispatch
    bids = {
        (i, k): pyo.value(model.x[i, k])
        for (i, k) in model.S
    }

    # net flows on each undirected edge
    flows = {}
    for (i, j) in model.U:
        f_plus  = pyo.value(model.f_plus[i, j])
        f_minus = pyo.value(model.f_minus[i, j])
        flows[(i, j)] = f_plus - f_minus

    return {
        'status': status,
        'cost': cost,
        'supply': supply,
        'import': imp,
        'export': exp,
        'flows': flows,
        'bids': bids,
    }
