from pyomo.environ import *
import json
import ast
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sns

class ModelVisualizer:
    @staticmethod
    def plot_demand_distribution(models_dict):
        # Extract data from models_dict
        model_data = models_dict.get('model_1', {})
        if not model_data:
            st.error("Model data not found in models_dict")
            return

        # Get stores and products from the data
        stores = model_data.get('Sets', {}).get('Stores', [])
        products = model_data.get('Sets', {}).get('Products', [])
        demand_data = model_data.get('Parameters', {}).get('Demand', {})

        # Debug information
        st.write("### Data Verification")
        st.write("Number of stores:", len(stores))
        st.write("Number of products:", len(products))
        st.write("Number of demand entries:", len(demand_data))
        st.write("Sample demand entry:", next(iter(demand_data.items())) if demand_data else "No demand data")

        if not stores or not products:
            st.error("Stores or Products data missing")
            return
        if not demand_data:
            st.error("Demand data missing")
            return

        try:
            # Create figure with larger size and better spacing
            plt.figure(figsize=(12, 6))
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Set up colors for better visibility
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            # Plot demand for each store
            for i, store in enumerate(stores):
                demands = []
                for product in products:
                    # Get demand value for store-product pair
                    demand = demand_data.get((store, product), 0)
                    demands.append(demand)
                    st.write(f"Debug - Store: {store}, Product: {product}, Demand: {demand}")
                
                # Plot with distinct color and style
                ax.plot(products, demands, marker='o', label=store, color=colors[i % len(colors)], 
                        linewidth=2, markersize=8)
            
            # Customize the plot
            ax.set_xlabel('Products', fontsize=12, labelpad=10)
            ax.set_ylabel('Demand', fontsize=12, labelpad=10)
            ax.set_title('Demand Distribution by Store', fontsize=14, pad=20)
            
            # Add grid and customize it
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Customize legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close(fig)

            # Print the data for verification
            st.write("### Demand Data Table")
            demand_table = []
            for store in stores:
                row = {'Store': store}
                for product in products:
                    row[product] = demand_data.get((store, product), 0)
                demand_table.append(row)
            st.table(demand_table)

        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.write("### Debug Information")
            st.write("Stores:", stores)
            st.write("Products:", products)
            st.write("Sample demand data:", dict(list(demand_data.items())[:5]))

    @staticmethod
    def plot_production_quantities(models_dict):
        # Extract data from models_dict
        model_data = models_dict.get('model_1', {})
        if not model_data:
            st.error("Model data not found in models_dict")
            return

        # Get fabs and products from the data
        fabs = model_data.get('Sets', {}).get('Fabs', [])
        products = model_data.get('Sets', {}).get('Products', [])
        fab_capacity = model_data.get('Parameters', {}).get('FabCapacity', {})

        if not fabs or not products:
            st.error("Fabs or Products data missing")
            return
        if not fab_capacity:
            st.error("Fab capacity data missing")
            return

        try:
            # Create figure with larger size
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Set the width of each bar and positions of the bars
            width = 0.35
            x = np.arange(len(fabs))
            
            # Plot capacity for each product
            for i, product in enumerate(products):
                capacities = []
                for fab in fabs:
                    capacity = fab_capacity.get((fab, product), 0)
                    capacities.append(capacity)
                
                # Plot bars with offset
                bars = ax.bar(x + i*width, capacities, width, 
                             label=f'{product.upper()} Capacity',
                             alpha=0.7)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom')
            
            # Customize the plot
            ax.set_xlabel('Manufacturing Facilities', fontsize=12, labelpad=10)
            ax.set_ylabel('Production Capacity', fontsize=12, labelpad=10)
            ax.set_title('Production Capacity by Facility and Product Type', fontsize=14, pad=20)
            
            # Set x-axis labels
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(fabs, rotation=45)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            # Customize legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust layout
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close(fig)

            # Create a summary table
            st.write("### Production Capacity Summary")
            summary_table = []
            total_cpu_capacity = 0
            total_gpu_capacity = 0
            
            for fab in fabs:
                row = {'Facility': fab}
                for product in products:
                    capacity = fab_capacity.get((fab, product), 0)
                    row[f'{product.upper()} Capacity'] = capacity
                    if product == 'cpu':
                        total_cpu_capacity += capacity
                    else:
                        total_gpu_capacity += capacity
                summary_table.append(row)
            
            # Add totals row
            summary_table.append({
                'Facility': 'Total',
                'CPU Capacity': total_cpu_capacity,
                'GPU Capacity': total_gpu_capacity
            })
            
            st.table(summary_table)

            # Show comparison with demand
            st.write("### Capacity vs Demand Analysis")
            demand_data = model_data.get('Parameters', {}).get('Demand', {})
            if demand_data:
                total_cpu_demand = sum(demand_data.get((store, 'cpu'), 0) 
                                     for store in model_data.get('Sets', {}).get('Stores', []))
                total_gpu_demand = sum(demand_data.get((store, 'gpu'), 0) 
                                     for store in model_data.get('Sets', {}).get('Stores', []))
                
                analysis_data = {
                    'Metric': ['Total Capacity', 'Total Demand', 'Capacity Utilization'],
                    'CPU': [
                        total_cpu_capacity,
                        total_cpu_demand,
                        f"{(total_cpu_demand/total_cpu_capacity*100):.1f}%"
                    ],
                    'GPU': [
                        total_gpu_capacity,
                        total_gpu_demand,
                        f"{(total_gpu_demand/total_gpu_capacity*100):.1f}%"
                    ]
                }
                st.table(analysis_data)

        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.write("### Debug Information")
            st.write("Fabs:", fabs)
            st.write("Products:", products)
            st.write("Sample capacity data:", dict(list(fab_capacity.items())[:5]))

    @staticmethod
    def plot_supply_chain_analysis(models_dict):
        # Extract data from models_dict
        model_data = models_dict.get('model_1', {})
        if not model_data:
            st.error("Model data not found in models_dict")
            return

        # Get all necessary data
        stores = model_data.get('Sets', {}).get('Stores', [])
        products = model_data.get('Sets', {}).get('Products', [])
        fabs = model_data.get('Sets', {}).get('Fabs', [])
        demand_data = model_data.get('Parameters', {}).get('Demand', {})
        fab_capacity = model_data.get('Parameters', {}).get('FabCapacity', {})
        trans_cost = model_data.get('Parameters', {}).get('TransCost', {})

        try:
            # 1. Capacity vs Demand Analysis
            st.write("## Capacity vs Demand Analysis")
            
            # Calculate totals
            total_cpu_demand = sum(demand_data.get((store, 'cpu'), 0) for store in stores)
            total_gpu_demand = sum(demand_data.get((store, 'gpu'), 0) for store in stores)
            total_cpu_capacity = sum(fab_capacity.get((fab, 'cpu'), 0) for fab in fabs)
            total_gpu_capacity = sum(fab_capacity.get((fab, 'gpu'), 0) for fab in fabs)

            # Create comparison plot
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(2)
            width = 0.35

            capacity_bars = ax.bar(x - width/2, [total_cpu_capacity, total_gpu_capacity], width, 
                                 label='Total Capacity', color='skyblue')
            demand_bars = ax.bar(x + width/2, [total_cpu_demand, total_gpu_demand], width, 
                               label='Total Demand', color='lightcoral')

            # Add value labels
            def autolabel(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom')

            autolabel(capacity_bars)
            autolabel(demand_bars)

            ax.set_ylabel('Units')
            ax.set_title('Capacity vs Demand by Product')
            ax.set_xticks(x)
            ax.set_xticklabels(['CPU', 'GPU'])
            ax.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Show utilization metrics
            st.write("### Capacity Utilization")
            utilization_data = {
                'Metric': ['CPU', 'GPU'],
                'Total Capacity': [f"{total_cpu_capacity:,}", f"{total_gpu_capacity:,}"],
                'Total Demand': [f"{total_cpu_demand:,}", f"{total_gpu_demand:,}"],
                'Utilization (%)': [
                    f"{(total_cpu_demand/total_cpu_capacity*100):.1f}%",
                    f"{(total_gpu_demand/total_gpu_capacity*100):.1f}%"
                ],
                'Excess Capacity': [
                    f"{(total_cpu_capacity - total_cpu_demand):,}",
                    f"{(total_gpu_capacity - total_gpu_demand):,}"
                ]
            }
            st.table(utilization_data)

            # 2. Transportation Cost Analysis
            st.write("## Transportation Cost Analysis")
            
            # Create heatmap of transportation costs
            trans_cost_matrix_cpu = np.zeros((len(fabs), len(stores)))
            trans_cost_matrix_gpu = np.zeros((len(fabs), len(stores)))

            for i, fab in enumerate(fabs):
                for j, store in enumerate(stores):
                    # Sum up costs through all possible assembly points
                    cpu_costs = []
                    gpu_costs = []
                    for assembly in model_data.get('Sets', {}).get('Assemblies', []):
                        cpu_key = (fab, assembly, 'cpu')
                        gpu_key = (fab, assembly, 'gpu')
                        if cpu_key in trans_cost:
                            cpu_costs.append(trans_cost[cpu_key])
                        if gpu_key in trans_cost:
                            gpu_costs.append(trans_cost[gpu_key])
                    
                    trans_cost_matrix_cpu[i, j] = min(cpu_costs) if cpu_costs else 0
                    trans_cost_matrix_gpu[i, j] = min(gpu_costs) if gpu_costs else 0

            # Plot heatmaps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # CPU Transportation Costs
            sns.heatmap(trans_cost_matrix_cpu, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1,
                       xticklabels=stores, yticklabels=fabs)
            ax1.set_title('CPU Transportation Costs')
            ax1.set_xlabel('Stores')
            ax1.set_ylabel('Fabs')

            # GPU Transportation Costs
            sns.heatmap(trans_cost_matrix_gpu, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax2,
                       xticklabels=stores, yticklabels=fabs)
            ax2.set_title('GPU Transportation Costs')
            ax2.set_xlabel('Stores')
            ax2.set_ylabel('Fabs')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Summary statistics
            st.write("### Transportation Cost Summary")
            cost_summary = {
                'Product': ['CPU', 'GPU'],
                'Average Cost': [
                    f"${trans_cost_matrix_cpu.mean():.2f}",
                    f"${trans_cost_matrix_gpu.mean():.2f}"
                ],
                'Min Cost': [
                    f"${trans_cost_matrix_cpu.min():.2f}",
                    f"${trans_cost_matrix_gpu.min():.2f}"
                ],
                'Max Cost': [
                    f"${trans_cost_matrix_cpu.max():.2f}",
                    f"${trans_cost_matrix_gpu.max():.2f}"
                ]
            }
            st.table(cost_summary)

        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.write("### Debug Information")
            st.write("Data available:", {
                "stores": len(stores),
                "products": len(products),
                "fabs": len(fabs),
                "demand_entries": len(demand_data),
                "capacity_entries": len(fab_capacity),
                "trans_cost_entries": len(trans_cost)
            })

class ModelOperations:
    @staticmethod
    def update_demand_and_resolve(model, store_name, demand_increase, data):
        # Update demand for the specified store
        for product in model.Products:
            if (store_name, product) in model.Demand:
                model.Demand[store_name, product] += demand_increase

        # Re-solve the model
        solver = SolverFactory('glpk')
        result = solver.solve(model, tee=True)

        if result.solver.termination_condition == TerminationCondition.optimal:
            # Update solution values
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

            # Recalculate and print financial breakdown
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

            st.write("\nUpdated Financial Breakdown:")
            st.write(f"Total Revenue: ${total_revenue:,.2f}")
            st.write(f"Total Production Cost: ${total_production_cost:,.2f}")
            st.write(f"Total Transportation Cost: ${total_transportation_cost:,.2f}")
            st.write(f"Total Assembly-to-Store Cost: ${total_assembly_to_store_cost:,.2f}")
            net_profit = total_revenue - total_production_cost - total_transportation_cost - total_assembly_to_store_cost
            st.write(f"Net Profit: ${net_profit:,.2f}")
            return True, model
        else:
            st.write("No optimal solution found.")
            return False, model

class SupplyChainAnalyzer:
    @staticmethod
    def analyze_store_profitability(models_dict):
        try:
            # Extract data
            stores = models_dict.get('model_1', {}).get('Sets', {}).get('Stores', [])
            products = models_dict.get('model_1', {}).get('Sets', {}).get('Products', [])
            demand_data = models_dict.get('model_1', {}).get('Parameters', {}).get('Demand', {})
            revenue_data = models_dict.get('model_1', {}).get('Parameters', {}).get('Revenue', {})

            # Calculate profitability per store
            store_profits = {}
            store_details = {}
            for store in stores:
                total_profit = 0
                product_profits = {}
                for product in products:
                    demand = demand_data.get((store, product), 0)
                    revenue = revenue_data.get((store, product), 0)
                    profit = demand * revenue
                    total_profit += profit
                    product_profits[product] = {
                        'demand': demand,
                        'revenue_per_unit': revenue,
                        'total_revenue': profit
                    }
                store_profits[store] = total_profit
                store_details[store] = product_profits

            # Sort stores by profitability
            sorted_profits = dict(sorted(store_profits.items(), key=lambda x: x[1], reverse=True))

            # Create analysis text
            analysis = "# Store Profitability Analysis\n\n"
            analysis += "## Overall Rankings\n"
            for rank, (store, profit) in enumerate(sorted_profits.items(), 1):
                analysis += f"{rank}. **{store}**: ${profit:,.2f}\n"
            
            analysis += "\n## Detailed Breakdown\n"
            for store, profit in sorted_profits.items():
                analysis += f"\n### {store} (Total: ${profit:,.2f})\n"
                for product, details in store_details[store].items():
                    analysis += f"- {product.upper()}:\n"
                    analysis += f"  * Demand: {details['demand']:,} units\n"
                    analysis += f"  * Revenue per unit: ${details['revenue_per_unit']:,.2f}\n"
                    analysis += f"  * Total revenue: ${details['total_revenue']:,.2f}\n"
            
            return analysis
        except Exception as e:
            return f"Error analyzing store profitability: {str(e)}"

    @staticmethod
    def analyze_price_optimization(models_dict):
        try:
            # Extract data
            stores = models_dict.get('model_1', {}).get('Sets', {}).get('Stores', [])
            products = models_dict.get('model_1', {}).get('Sets', {}).get('Products', [])
            revenue_data = models_dict.get('model_1', {}).get('Parameters', {}).get('Revenue', {})

            # Calculate average prices per product
            price_analysis = {}
            for product in products:
                prices = [revenue_data.get((store, product), 0) for store in stores]
                avg_price = sum(prices) / len(prices)
                price_analysis[product] = {
                    'average': avg_price,
                    'min': min(prices),
                    'max': max(prices),
                    'opportunities': []
                }

                # Identify pricing opportunities
                for store in stores:
                    price = revenue_data.get((store, product), 0)
                    if price < avg_price * 0.95:  # If price is more than 5% below average
                        price_analysis[product]['opportunities'].append({
                            'store': store,
                            'current_price': price,
                            'suggested_price': avg_price,
                            'potential_increase': avg_price - price
                        })

            # Create analysis text
            analysis = "Price Optimization Opportunities:\n\n"
            for product, data in price_analysis.items():
                analysis += f"{product.upper()} Analysis:\n"
                analysis += f"Average Price: ${data['average']:,.2f}\n"
                analysis += f"Price Range: ${data['min']:,.2f} - ${data['max']:,.2f}\n\n"
                
                if data['opportunities']:
                    analysis += "Recommended Price Adjustments:\n"
                    for opp in data['opportunities']:
                        analysis += f"- {opp['store']}: Increase from ${opp['current_price']:,.2f} to ${opp['suggested_price']:,.2f}\n"
                        analysis += f"  Potential Revenue Increase: ${opp['potential_increase']:,.2f} per unit\n"
                analysis += "\n"
            
            return analysis
        except Exception as e:
            return f"Error analyzing price optimization: {str(e)}"

    @staticmethod
    def analyze_transportation_optimization(models_dict):
        try:
            # Extract data
            fabs = models_dict.get('model_1', {}).get('Sets', {}).get('Fabs', [])
            assemblies = models_dict.get('model_1', {}).get('Sets', {}).get('Assemblies', [])
            stores = models_dict.get('model_1', {}).get('Sets', {}).get('Stores', [])
            products = models_dict.get('model_1', {}).get('Sets', {}).get('Products', [])
            trans_cost = models_dict.get('model_1', {}).get('Parameters', {}).get('TransCost', {})

            analysis = "# Transportation Cost Analysis\n\n"
            
            for product in products:
                analysis += f"## {product.upper()} Transportation Routes\n\n"
                
                # Analyze Fab to Assembly costs
                analysis += "### Manufacturing to Assembly Transportation:\n"
                min_fab_cost = float('inf')
                max_fab_cost = 0
                min_fab_route = None
                max_fab_route = None
                
                for fab in fabs:
                    for assembly in assemblies:
                        cost = trans_cost.get((fab, assembly, product), None)
                        if cost is not None and cost > 0:  # Only consider valid costs
                            route = f"{fab} → {assembly}"
                            if cost < min_fab_cost:
                                min_fab_cost = cost
                                min_fab_route = route
                            if cost > max_fab_cost:
                                max_fab_cost = cost
                                max_fab_route = route
                
                if min_fab_route and max_fab_route:
                    analysis += f"- Most economical route: {min_fab_route} (${min_fab_cost:,.2f} per unit)\n"
                    analysis += f"- Most expensive route: {max_fab_route} (${max_fab_cost:,.2f} per unit)\n"
                    analysis += f"- Potential savings per unit: ${(max_fab_cost - min_fab_cost):,.2f}\n"
                else:
                    analysis += "No valid transportation routes found between manufacturing and assembly.\n"
                
                # Analyze Assembly to Store costs
                analysis += "\n### Assembly to Store Transportation:\n"
                min_store_cost = float('inf')
                max_store_cost = 0
                min_store_route = None
                max_store_route = None
                
                for assembly in assemblies:
                    for store in stores:
                        cost = trans_cost.get((assembly, store, product), None)
                        if cost is not None and cost > 0:  # Only consider valid costs
                            route = f"{assembly} → {store}"
                            if cost < min_store_cost:
                                min_store_cost = cost
                                min_store_route = route
                            if cost > max_store_cost:
                                max_store_cost = cost
                                max_store_route = route
                
                if min_store_route and max_store_route:
                    analysis += f"- Most economical route: {min_store_route} (${min_store_cost:,.2f} per unit)\n"
                    analysis += f"- Most expensive route: {max_store_route} (${max_store_cost:,.2f} per unit)\n"
                    analysis += f"- Potential savings per unit: ${(max_store_cost - min_store_cost):,.2f}\n"
                else:
                    analysis += "No valid transportation routes found between assembly and stores.\n"
                
                # Total end-to-end analysis
                if min_fab_cost != float('inf') and min_store_cost != float('inf'):
                    min_total = min_fab_cost + min_store_cost
                    max_total = max_fab_cost + max_store_cost
                    analysis += f"\n### Total End-to-End Transportation:\n"
                    analysis += f"- Minimum total cost: ${min_total:,.2f} per unit\n"
                    analysis += f"- Maximum total cost: ${max_total:,.2f} per unit\n"
                    analysis += f"- Maximum potential savings: ${(max_total - min_total):,.2f} per unit\n"
                
                analysis += "\n"
            
            return analysis
        except Exception as e:
            return f"Error analyzing transportation optimization: {str(e)}"

    @staticmethod
    def analyze_production_allocation(models_dict):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            # Extract data
            fabs = models_dict.get('model_1', {}).get('Sets', {}).get('Fabs', [])
            products = models_dict.get('model_1', {}).get('Sets', {}).get('Products', [])
            fab_capacity = models_dict.get('model_1', {}).get('Parameters', {}).get('FabCapacity', {})
            demand_data = models_dict.get('model_1', {}).get('Parameters', {}).get('Demand', {})

            # Calculate total demand and capacity
            analysis = "# Production and Capacity Analysis\n\n"
            
            # Create subplots for visualizations
            fig = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 2, figure=fig)
            ax1 = fig.add_subplot(gs[0, :])  # Capacity vs Demand
            ax2 = fig.add_subplot(gs[1, 0])  # CPU Utilization
            ax3 = fig.add_subplot(gs[1, 1])  # GPU Utilization

            # Colors for visualization
            colors = {'cpu': '#2ecc71', 'gpu': '#3498db'}
            
            # Data for overall capacity vs demand
            cap_demand_data = {'Product': [], 'Metric': [], 'Value': []}
            
            for product in products:
                total_demand = sum(demand_data.get((store, product), 0) 
                                 for store in models_dict['model_1']['Sets']['Stores'])
                total_capacity = sum(fab_capacity.get((fab, product), 0) for fab in fabs)
                
                # Add to plotting data
                cap_demand_data['Product'].extend([product.upper()] * 2)
                cap_demand_data['Metric'].extend(['Capacity', 'Demand'])
                cap_demand_data['Value'].extend([total_capacity, total_demand])
                
                analysis += f"\n## {product.upper()}\n"
                analysis += f"Total Demand: {total_demand:,} units\n"
                analysis += f"Total Capacity: {total_capacity:,} units\n"
                
                # Sort fabs by capacity
                fab_caps = [(fab, fab_capacity.get((fab, product), 0)) for fab in fabs]
                fab_caps.sort(key=lambda x: x[1], reverse=True)
                
                analysis += "Recommended Production Allocation:\n"
                remaining_demand = total_demand
                
                # Data for utilization charts
                util_data = []
                fab_names = []
                
                for fab, cap in fab_caps:
                    if remaining_demand <= 0:
                        allocation = 0
                        status = "Idle"
                    elif remaining_demand >= cap:
                        allocation = cap
                        status = "Full Capacity"
                    else:
                        allocation = remaining_demand
                        status = "Partial Capacity"
                    
                    remaining_demand -= allocation
                    utilization = (allocation / cap * 100) if cap > 0 else 0
                    
                    analysis += f"- {fab}: {allocation:,} units ({utilization:.1f}% utilization) - {status}\n"
                    
                    # Add to utilization data
                    util_data.append(utilization)
                    fab_names.append(fab)
                
                # Plot utilization for this product
                ax = ax2 if product == 'cpu' else ax3
                bars = ax.bar(fab_names, util_data, color=colors[product], alpha=0.7)
                ax.set_title(f'{product.upper()} Capacity Utilization by Facility')
                ax.set_ylabel('Utilization (%)')
                ax.set_ylim(0, 110)  # Leave some space above 100%
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom')
                
                ax.tick_params(axis='x', rotation=45)
            
            # Plot overall capacity vs demand
            x = np.arange(len(products))
            width = 0.35
            
            cap_values = [v for p, m, v in zip(cap_demand_data['Product'], cap_demand_data['Metric'], cap_demand_data['Value']) if m == 'Capacity']
            dem_values = [v for p, m, v in zip(cap_demand_data['Product'], cap_demand_data['Metric'], cap_demand_data['Value']) if m == 'Demand']
            
            bars1 = ax1.bar(x - width/2, cap_values, width, label='Total Capacity', color='lightblue')
            bars2 = ax1.bar(x + width/2, dem_values, width, label='Total Demand', color='lightcoral')
            
            ax1.set_ylabel('Units')
            ax1.set_title('Capacity vs Demand by Product')
            ax1.set_xticks(x)
            ax1.set_xticklabels([p.upper() for p in products])
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels
            def autolabel(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom')
            
            autolabel(bars1)
            autolabel(bars2)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            return analysis
        except Exception as e:
            return f"Error analyzing production allocation: {str(e)}"

class SupplyChainVisualizer:
    @staticmethod
    def create_facility_heatmap(models_dict):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            # Extract data
            fabs = models_dict.get('model_1', {}).get('Sets', {}).get('Fabs', [])
            products = models_dict.get('model_1', {}).get('Sets', {}).get('Products', [])
            fab_capacity = models_dict.get('model_1', {}).get('Parameters', {}).get('FabCapacity', {})
            
            # Create capacity matrix
            capacity_matrix = np.zeros((len(fabs), len(products)))
            for i, fab in enumerate(fabs):
                for j, product in enumerate(products):
                    capacity_matrix[i, j] = fab_capacity.get((fab, product), 0)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(capacity_matrix, 
                       annot=True, 
                       fmt='.0f',  # Changed from ',d' to '.0f' to handle float values
                       cmap='YlOrRd',
                       xticklabels=[p.upper() for p in products], 
                       yticklabels=fabs,
                       cbar_kws={'label': 'Capacity Units'})
            
            plt.title('Manufacturing Facility Capacity Distribution', pad=20)
            plt.xlabel('Products', labelpad=10)
            plt.ylabel('Facilities', labelpad=10)
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()

            # Add capacity analysis
            total_cpu_capacity = sum(fab_capacity.get((fab, 'cpu'), 0) for fab in fabs)
            total_gpu_capacity = sum(fab_capacity.get((fab, 'gpu'), 0) for fab in fabs)
            
            stores = models_dict.get('model_1', {}).get('Sets', {}).get('Stores', [])
            demand_data = models_dict.get('model_1', {}).get('Parameters', {}).get('Demand', {})
            total_cpu_demand = sum(demand_data.get((store, 'cpu'), 0) for store in stores)
            total_gpu_demand = sum(demand_data.get((store, 'gpu'), 0) for store in stores)

            st.write("\n### Capacity Analysis")
            st.write(f"**CPU Capacity vs Demand:**")
            st.write(f"- Total Capacity: {total_cpu_capacity:,.0f} units")
            st.write(f"- Total Demand: {total_cpu_demand:,.0f} units")
            if total_cpu_capacity > 0:
                st.write(f"- Utilization: {(total_cpu_demand/total_cpu_capacity*100):.1f}%")
            else:
                st.write("- Utilization: N/A (No capacity available)")
            
            st.write(f"\n**GPU Capacity vs Demand:**")
            st.write(f"- Total Capacity: {total_gpu_capacity:,.0f} units")
            st.write(f"- Total Demand: {total_gpu_demand:,.0f} units")
            if total_gpu_capacity > 0:
                st.write(f"- Utilization: {(total_gpu_demand/total_gpu_capacity*100):.1f}%")
            else:
                st.write("- Utilization: N/A (No capacity available)")

        except Exception as e:
            st.error(f"Error creating facility heatmap: {str(e)}")
            # Print debug information
            st.write("Debug Information:")
            st.write("Fabs:", fabs)
            st.write("Products:", products)
            st.write("Sample capacity data:", dict(list(fab_capacity.items())[:5]))

    @staticmethod
    def create_utilization_sunburst(models_dict):
        try:
            import plotly.graph_objects as go
            
            # Extract data
            fabs = models_dict.get('model_1', {}).get('Sets', {}).get('Fabs', [])
            products = models_dict.get('model_1', {}).get('Sets', {}).get('Products', [])
            fab_capacity = models_dict.get('model_1', {}).get('Parameters', {}).get('FabCapacity', {})
            demand_data = models_dict.get('model_1', {}).get('Parameters', {}).get('Demand', {})

            # Calculate utilization
            labels = ['Total']
            parents = ['']
            values = [0]
            
            total_value = 0
            for product in products:
                total_demand = sum(demand_data.get((store, product), 0) 
                                 for store in models_dict['model_1']['Sets']['Stores'])
                total_capacity = sum(fab_capacity.get((fab, product), 0) for fab in fabs)
                
                # Add product level
                labels.append(f'{product.upper()}')
                parents.append('Total')
                values.append(total_capacity)
                total_value += total_capacity
                
                # Add utilization breakdown
                labels.extend([f'{product.upper()} Used', f'{product.upper()} Unused'])
                parents.extend([f'{product.upper()}', f'{product.upper()}'])
                values.extend([total_demand, total_capacity - total_demand])
            
            values[0] = total_value
            
            fig = go.Figure(go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
            ))
            fig.update_layout(width=800, height=800, title="Capacity Utilization Sunburst")
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating utilization sunburst: {str(e)}")

    @staticmethod
    def create_sankey_diagram(models_dict):
        try:
            import plotly.graph_objects as go
            
            # Extract data
            fabs = models_dict.get('model_1', {}).get('Sets', {}).get('Fabs', [])
            products = models_dict.get('model_1', {}).get('Sets', {}).get('Products', [])
            stores = models_dict.get('model_1', {}).get('Sets', {}).get('Stores', [])
            fab_capacity = models_dict.get('model_1', {}).get('Parameters', {}).get('FabCapacity', {})
            demand_data = models_dict.get('model_1', {}).get('Parameters', {}).get('Demand', {})

            # Create nodes
            nodes = []
            node_labels = []
            
            # Add fabs
            for fab in fabs:
                nodes.append(fab)
                node_labels.append(fab)
            
            # Add products
            for product in products:
                nodes.append(f"{product}_production")
                node_labels.append(f"{product.upper()} Production")
            
            # Add stores
            for store in stores:
                nodes.append(store)
                node_labels.append(store)

            # Create links
            source = []
            target = []
            value = []
            
            # Fab to product links
            for i, fab in enumerate(fabs):
                for j, product in enumerate(products):
                    capacity = fab_capacity.get((fab, product), 0)
                    if capacity > 0:
                        source.append(nodes.index(fab))
                        target.append(nodes.index(f"{product}_production"))
                        value.append(capacity)
            
            # Product to store links
            for product in products:
                prod_idx = nodes.index(f"{product}_production")
                for store in stores:
                    demand = demand_data.get((store, product), 0)
                    if demand > 0:
                        source.append(prod_idx)
                        target.append(nodes.index(store))
                        value.append(demand)

            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = "black", width = 0.5),
                    label = node_labels,
                    color = "blue"
                ),
                link = dict(
                    source = source,
                    target = target,
                    value = value
                )
            )])

            fig.update_layout(title_text="Supply Chain Flow", font_size=10, height=800)
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating Sankey diagram: {str(e)}")

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

# Function to handle visualization requests
def handle_visualization_request(viz_type, models_dict):
    visualizer = SupplyChainVisualizer()
    model_visualizer = ModelVisualizer()

    if viz_type == "heatmap":
        st.write("### Facility Capacity Heatmap")
        visualizer.create_facility_heatmap(models_dict)
        return "Showing facility capacity distribution heatmap."

    elif viz_type == "demand":
        st.write("### Demand Distribution Analysis")
        model_visualizer.plot_demand_distribution(models_dict)
        return "Showing demand distribution across stores and products."

    elif viz_type == "production":
        st.write("### Production and Capacity Analysis")
        model_visualizer.plot_production_quantities(models_dict)
        return "Showing production allocation and capacity utilization."

    elif viz_type == "transportation":
        st.write("### Transportation Cost Analysis")
        model_visualizer.plot_supply_chain_analysis(models_dict)
        return "Showing transportation cost analysis and optimization opportunities."

    elif viz_type == "sunburst":
        st.write("### Capacity Utilization Sunburst")
        visualizer.create_utilization_sunburst(models_dict)
        return "Showing capacity utilization sunburst diagram."

    elif viz_type == "flow":
        st.write("### Supply Chain Flow Diagram")
        visualizer.create_sankey_diagram(models_dict)
        return "Showing supply chain flow Sankey diagram."

    elif viz_type == "all":
        st.write("### Comprehensive Supply Chain Visualization")
        st.write("#### 1. Facility Capacity Heatmap")
        visualizer.create_facility_heatmap(models_dict)
        st.write("#### 2. Demand Distribution")
        model_visualizer.plot_demand_distribution(models_dict)
        st.write("#### 3. Production Analysis")
        model_visualizer.plot_production_quantities(models_dict)
        st.write("#### 4. Transportation Analysis")
        model_visualizer.plot_supply_chain_analysis(models_dict)
        st.write("#### 5. Capacity Utilization Sunburst")
        visualizer.create_utilization_sunburst(models_dict)
        st.write("#### 6. Supply Chain Flow")
        visualizer.create_sankey_diagram(models_dict)
        return "Showing comprehensive supply chain analysis."

    else:
        return "Available visualizations:\n" + \
               "1. 'Show capacity heatmap'\n" + \
               "2. 'Show demand distribution'\n" + \
               "3. 'Show production allocation'\n" + \
               "4. 'Show transportation costs'\n" + \
               "5. 'Show utilization sunburst'\n" + \
               "6. 'Show supply chain flow'\n" + \
               "7. 'Show all visualizations'\n" + \
               "Please try one of these requests."

# Function to handle model operations
def handle_model_operations(request, model, data):
    operations = ModelOperations()
    
    if "increase demand" in request.lower():
        # Extract store and amount from request
        # This is a simple example; you might want to use more sophisticated parsing
        store_name = "store2"  # Default or parse from request
        amount = 20  # Default or parse from request
        
        success, updated_model = operations.update_demand_and_resolve(model, store_name, amount, data)
        if success:
            return "Demand updated and model resolved successfully.", updated_model
        else:
            return "Failed to update demand and resolve model.", model
    
    return "Operation not recognized.", model

# Function to handle analysis requests
def handle_analysis_request(request, models_dict):
    analyzer = SupplyChainAnalyzer()
    request = request.lower()
    
    analysis_types = {
        "profitability": (
            ["profitability", "profit", "revenue", "store performance"],
            analyzer.analyze_store_profitability
        ),
        "pricing": (
            ["price", "pricing", "revenue optimization", "price optimization"],
            analyzer.analyze_price_optimization
        ),
        "transportation": (
            ["transportation", "shipping", "logistics", "routing"],
            analyzer.analyze_transportation_optimization
        ),
        "production": (
            ["production", "allocation", "capacity", "manufacturing"],
            analyzer.analyze_production_allocation
        )
    }
    
    responses = []
    for analysis_type, (keywords, analysis_func) in analysis_types.items():
        if any(keyword in request for keyword in keywords):
            try:
                analysis_result = analysis_func(models_dict)
                responses.append(analysis_result)
            except Exception as e:
                responses.append(f"Error analyzing {analysis_type}: {str(e)}")
    
    if not responses:
        return "I can analyze:\n" + \
               "1. Store profitability (e.g., 'Show store profitability')\n" + \
               "2. Price optimization (e.g., 'Analyze pricing opportunities')\n" + \
               "3. Transportation optimization (e.g., 'Optimize transportation costs')\n" + \
               "4. Production allocation (e.g., 'Show production allocation')\n" + \
               "Please try one of these requests."
    
    return "\n\n".join(responses)

# End of file - remove any code after this point that tries to create or run the model