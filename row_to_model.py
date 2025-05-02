import json

# Load your JSON files
with open('demands.json') as f:
    demands = json.load(f)
with open('processing.json') as f:
    processing = json.load(f)
with open('transportation.json') as f:
    transportation = json.load(f)

# Build sets
stores = sorted(set(row['node_id'] for row in demands))
products = sorted(set(row['product_type'] for row in demands))
fabs = sorted(set(row['node_id'] for row in processing))
assemblies = sorted(set(row['destination'] for row in transportation))

# Build parameters
Demand = {f"('{row['node_id']}', '{row['product_type']}')": int(row['demand']) for row in demands}
Revenue = {f"('{row['node_id']}', '{row['product_type']}')": int(row['revenue_per_unit']) for row in demands}
FabCapacity = {f"('{row['node_id']}', '{row['product_type']}')": int(row['capacity']) for row in processing}
FabCost = {f"('{row['node_id']}', '{row['product_type']}')": int(row['cost_per_unit']) for row in processing}
TransCost = {f"('{row['origin']}', '{row['destination']}', '{row['product_type']}')": int(row['cost_per_unit']) for row in transportation}
TransCapacity = {f"('{row['origin']}', '{row['destination']}', '{row['product_type']}')": int(row['capacity']) for row in transportation}

# Combine into one dictionary
model_data = {
    "Stores": stores,
    "Products": products,
    "Fabs": fabs,
    "Assemblies": assemblies,
    "Demand": Demand,
    "Revenue": Revenue,
    "FabCapacity": FabCapacity,
    "FabCost": FabCost,
    "TransCost": TransCost,
    "TransCapacity": TransCapacity
}

# Save to a single JSON file
with open('supply_chain_data.json', 'w') as f:
    json.dump(model_data, f, indent=4)