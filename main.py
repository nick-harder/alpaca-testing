# run_scenarios_local.py (fixed undirected flows and clean printing)
# ----------------------
# Loads scenarios and calls build_model + solve_model, gathering results for analysis

import yaml
from market_clearing import solve_model
from market_clearing import build_model


def load_scenarios(path='test_scenarios.yaml'):
    """
    Read YAML file with a top-level 'cases' list. Each case must contain:
      name, nodes, line_cap (keys "i,j"), demand, bids.
    Returns a list of scenarios (currently only the first is used).
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    cases = []
    for c in data.get('cases', []):
        # normalize nodes
        nodes = [str(n).strip() for n in c.get('nodes', [])]

        # parse and normalize undirected line capacities
        raw_caps = c.get('line_cap', {})
        line_cap = {}
        for key, cap in raw_caps.items():
            i, j = [x.strip() for x in key.split(',')]
            edge = tuple(sorted((i, j)))
            line_cap[edge] = float(cap)

        # parse demands
        demand = {str(n).strip(): float(d) for n, d in c.get('demand', {}).items()}

        # parse bids
        bids = {
            str(node).strip(): [(float(q), float(p)) for q, p in segs]
            for node, segs in c.get('bids', {}).items()
        }

        cases.append({
            'name':     c.get('name', '').strip(),
            'nodes':    nodes,
            'line_cap': line_cap,
            'demand':   demand,
            'bids':     bids
        })
        break   # only load the first scenario for now
    return cases


def print_scenario_info(cfg):
    """Print the raw scenario configuration."""
    print(f"Scenario '{cfg['name']}' configuration:")
    print(f"  Nodes   : {cfg['nodes']}")
    print("  Demands :")
    for node, D in cfg['demand'].items():
        print(f"    Node {node}: D = {D}")
    print("  Line capacities:")
    seen = set()
    for (i, j), cap in cfg['line_cap'].items():
        u, v = tuple(sorted((i, j)))
        if (u, v) in seen:
            continue
        print(f"    {u}↔{v}: cap = {cap}")
        seen.add((u, v))
    print("  Bids (quantity, price) by node:")
    for node, segs in cfg['bids'].items():
        print(f"    Node {node}:")
        for k, (q, p) in enumerate(segs):
            print(f"      Segment {k}: q = {q}, p = {p}")
    print()


def print_results(cfg, variant, result):
    """Print solution status, cost, node‐by‐node balances, undirected flows, and bids."""
    print(f"--- {variant.upper():15s} ---")
    print(f"Status : {result['status']}")
    print(f"Cost   : {result['cost']:.2f} €\n")

    # Directly use the undirected flows as returned by the solver
    # (keys are sorted tuples)
    flows_ud = {tuple(sorted(k)): v for k, v in result['flows'].items()}

    # NODE‐BY‐NODE BALANCE
    print("Node results:")
    for node in cfg['nodes']:
        gen = result['supply'][node]
        incoming = 0.0
        outgoing = 0.0
        for (u, v), f in flows_ud.items():
            if node == u:
                # positive f is u->v
                outgoing += max(f, 0)
                incoming += max(-f, 0)
            elif node == v:
                # positive f is u->v into v
                incoming += max(f, 0)
                outgoing += max(-f, 0)
        print(
            f"  Node {node:>2s}: gen={gen:7.2f}, in={incoming:7.2f}, out={outgoing:7.2f}, "
            f"demand={cfg['demand'][node]:7.2f}"
        )

    # UNDIRECTED FLOWS
    print("\nFlows (undirected):")
    for (u, v), f in sorted(flows_ud.items()):
        print(f"  {u}↔{v}: {f:7.2f}")

    # BID ACCEPTANCES
    print("\nBid acceptances:")
    for node in cfg['nodes']:
        print(f"  Node {node} bids:")
        for k, (q, p) in enumerate(cfg['bids'][node]):
            accepted = result['bids'][(node, k)]
            print(f"    Seg{k}: offered={q:.2f}@{p:.2f} → accepted={accepted:.2f}")
    print()

# %%
scenarios = load_scenarios()
for cfg in scenarios:
    # print_scenario_info(cfg)
    for variant in ['baseline']:
        model = build_model(cfg, 
                            allow_simultaneous_import_export=False,
                            )
        result = solve_model(model)
        print_results(cfg, variant, result)

# %%
