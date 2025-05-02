from pyomo.environ import *
import json
import ast


def create_model(data):
    model = ConcreteModel()

    # Convert string tuples to actual tuples in the data
    def convert_string_tuples_to_tuples(d):
        return {ast.literal_eval(k): v for k, v in d.items()} if isinstance(d, dict) else d

    # Convert all dictionary data
    for key in ['Demand', 'Revenue', 'FabCapacity', 'FabCost', 'TransCost', 'TransCapacity']:
        if key in data:
            data[key] = convert_string_tuples_to_tuples(data[key])

    # Sets
    model.Products = Set(initialize=data['Products'])
    model.Stores = Set(initialize=data['Stores'])
    model.Fabs = Set(initialize=[f for f in data['Fabs'] if f.startswith('fab')])  # Only actual fabs
    model.Assemblies = Set(initialize=[a for a in data['Assemblies'] if a.startswith('assembly')])  # Only assemblies

    # Parameters
    def demand_init(m, s, p):
        return data['Demand'].get((s, p), 0)
    model.Demand = Param(model.Stores, model.Products, initialize=demand_init, default=0)

    def revenue_init(m, s, p):
        return data['Revenue'].get((s, p), 0)
    model.Revenue = Param(model.Stores, model.Products, initialize=revenue_init, default=0)

    def fab_capacity_init(m, f, p):
        return data['FabCapacity'].get((f, p), 0)
    model.FabCapacity = Param(model.Fabs, model.Products, initialize=fab_capacity_init, default=0)

    def fab_cost_init(m, f, p):
        return data['FabCost'].get((f, p), 0)
    model.FabCost = Param(model.Fabs, model.Products, initialize=fab_cost_init, default=0)

    def trans_cost_init(m, f, a, p):
        return data['TransCost'].get((f, a, p), 0)
    model.TransCost = Param(model.Fabs, model.Assemblies, model.Products, initialize=trans_cost_init, default=0)

    def trans_capacity_init(m, f, a, p):
        return data['TransCapacity'].get((f, a, p), float('inf'))  # Default to unconstrained if not specified
    model.TransCapacity = Param(model.Fabs, model.Assemblies, model.Products, initialize=trans_capacity_init, default=float('inf'))

    # Variables
    model.Prod = Var(model.Fabs, model.Products, domain=NonNegativeReals)
    model.Trans = Var(model.Fabs, model.Assemblies, model.Products, domain=NonNegativeReals)
    model.AssemblyToStore = Var(model.Assemblies, model.Stores, model.Products, domain=NonNegativeReals)

    # Objective: Maximize profit
    def profit_rule(m):
        revenue = sum(m.Revenue[s, p] * m.Demand[s, p] for s in m.Stores for p in m.Products)
        prod_cost = sum(m.FabCost[f, p] * m.Prod[f, p] for f in m.Fabs for p in m.Products)
        trans_cost = sum(m.TransCost[f, a, p] * m.Trans[f, a, p] 
                        for f in m.Fabs for a in m.Assemblies for p in m.Products)
        assembly_to_store_cost = sum(data['TransCost'].get((a, s, p), 0) * m.AssemblyToStore[a, s, p]
                                   for a in m.Assemblies for s in m.Stores for p in m.Products)
        return revenue - prod_cost - trans_cost - assembly_to_store_cost
    model.Profit = Objective(rule=profit_rule, sense=maximize)

    # Constraints
    def fab_capacity_rule(m, f, p):
        return m.Prod[f, p] <= m.FabCapacity[f, p]
    model.FabCapacityConstr = Constraint(model.Fabs, model.Products, rule=fab_capacity_rule)

    def trans_capacity_rule(m, f, a, p):
        return m.Trans[f, a, p] <= m.TransCapacity[f, a, p]
    model.TransCapacityConstr = Constraint(model.Fabs, model.Assemblies, model.Products, rule=trans_capacity_rule)

    def flow_balance_fab_rule(m, f, p):
        return sum(m.Trans[f, a, p] for a in m.Assemblies) == m.Prod[f, p]
    model.FlowBalanceFab = Constraint(model.Fabs, model.Products, rule=flow_balance_fab_rule)

    def flow_balance_assembly_rule(m, a, p):
        inflow = sum(m.Trans[f, a, p] for f in m.Fabs)
        outflow = sum(m.AssemblyToStore[a, s, p] for s in m.Stores)
        return inflow == outflow
    model.FlowBalanceAssembly = Constraint(model.Assemblies, model.Products, rule=flow_balance_assembly_rule)

    def demand_satisfaction_rule(m, s, p):
        return sum(m.AssemblyToStore[a, s, p] for a in m.Assemblies) >= m.Demand[s, p]
    model.DemandSatisfaction = Constraint(model.Stores, model.Products, rule=demand_satisfaction_rule)

    # Modified solver configuration
    available_solvers = ['cbc', 'ipopt', 'glpk', 'gurobi']
    solver = None
    
    for solver_name in available_solvers:
        try:
            solver = SolverFactory(solver_name)
            if solver.available():
                print(f"Using {solver_name} solver")
                break
        except:
            continue
    
    if solver is None:
        raise Exception("No available solvers found. Please install one of: CBC, IPOPT, GLPK, or Gurobi")

    results = solver.solve(model)

    # Store solution values in the variables
    for f in model.Fabs:
        for p in model.Products:
            model.Prod[f, p].value = value(model.Prod[f, p])
    
    for f in model.Fabs:
        for a in model.Assemblies:
            for p in model.Products:
                model.Trans[f, a, p].value = value(model.Trans[f, a, p])
    
    for a in model.Assemblies:
        for s in model.Stores:
            for p in model.Products:
                model.AssemblyToStore[a, s, p].value = value(model.AssemblyToStore[a, s, p])

    # Calculate and store all cost components
    total_revenue = sum(model.Revenue[s, p] * model.Demand[s, p] 
                       for s in model.Stores for p in model.Products)
    
    total_production_cost = sum(model.FabCost[f, p] * value(model.Prod[f, p]) 
                              for f in model.Fabs for p in model.Products)
    
    total_transportation_cost = sum(model.TransCost[f, a, p] * value(model.Trans[f, a, p])
                                  for f in model.Fabs 
                                  for a in model.Assemblies 
                                  for p in model.Products)
    
    total_assembly_to_store_cost = sum(data['TransCost'].get((a, s, p), 0) * value(model.AssemblyToStore[a, s, p])
                                     for a in model.Assemblies 
                                     for s in model.Stores 
                                     for p in model.Products)

    # Store cost components in a dictionary for easy access
    model.cost_components = {
        'total_revenue': float(value(total_revenue)),
        'total_production_cost': float(value(total_production_cost)),
        'total_transportation_cost': float(value(total_transportation_cost)),
        'total_assembly_to_store_cost': float(value(total_assembly_to_store_cost)),
        'net_profit': float(value(model.Profit))
    }

    # Store solution values in easily accessible format
    model.solution = {
        'costs': model.cost_components,
        'production': {(str(f), str(p)): float(value(model.Prod[f, p])) 
                      for f in model.Fabs for p in model.Products 
                      if value(model.Prod[f, p]) > 0},
        'transportation': {(str(f), str(a), str(p)): float(value(model.Trans[f, a, p])) 
                         for f in model.Fabs for a in model.Assemblies for p in model.Products 
                         if value(model.Trans[f, a, p]) > 0},
        'assembly_to_store': {(str(a), str(s), str(p)): float(value(model.AssemblyToStore[a, s, p])) 
                            for a in model.Assemblies for s in model.Stores for p in model.Products 
                            if value(model.AssemblyToStore[a, s, p]) > 0}
    }

    # Print detailed financial breakdown
    print("\nFinancial Breakdown:")
    print(f"Total Revenue: ${model.cost_components['total_revenue']:,.2f}")
    print(f"Total Production Cost: ${model.cost_components['total_production_cost']:,.2f}")
    print(f"Total Transportation Cost: ${model.cost_components['total_transportation_cost']:,.2f}")
    print(f"Total Assembly-to-Store Cost: ${model.cost_components['total_assembly_to_store_cost']:,.2f}")
    print(f"Net Profit: ${model.cost_components['net_profit']:,.2f}")
    total_cost = (model.cost_components['total_production_cost'] + 
                 model.cost_components['total_transportation_cost'] + 
                 model.cost_components['total_assembly_to_store_cost'])
    print(f"\nTotal Cost: ${total_cost:,.2f}")

    return model

# Save this as supply_chain_model.py