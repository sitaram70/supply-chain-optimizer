import streamlit as st
from openai import OpenAI
import os
from io import StringIO
import time
import tempfile
import io
from extractor import initial_loading
from extractor import update_model_representation, get_skipJSON, feed_skipJSON
from utils import get_agents
from utils import OptiChat_workflow_exp
from pyomo.opt import TerminationCondition
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Feas.supply_chain_model import (
    create_model, 
    handle_visualization_request, 
    handle_model_operations,
    handle_analysis_request,
    SupplyChainAnalyzer,
    SupplyChainVisualizer
)


def string_generator(long_string, chunk_size=50):
    for i in range(0, len(long_string), chunk_size):
        yield long_string[i:i+chunk_size]
        time.sleep(0.1)  # Optionally add a small delay between each yield


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
st.session_state['client'] = client
st.session_state['temperature'] = 0.1  # by default
st.session_state['json_mode'] = True  # by default
st.session_state['illustration_stream'] = True  # by default
st.session_state['inference_stream'] = True  # by default
st.session_state['explanation_stream'] = True  # by default
st.session_state['internal_experiment'] = False  # by default
st.session_state['external_experiment'] = False  # by default

st.set_page_config(layout='wide')

st.title("OptiChat: Talk to your Optimization Model")


gpt_model = st.sidebar.selectbox(label="GPT-Model", options=["gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], )
st.session_state["gpt_model"] = gpt_model
# Set a default model
if "gpt_model" not in st.session_state:
    st.session_state["gpt_model"] = "gpt-4-turbo-preview"
if "models_dict" not in st.session_state:
    st.session_state["models_dict"] = {"model_representation": {}}
if "code" not in st.session_state:
    st.session_state["code"] = ""

st.sidebar.subheader("Load Pyomo File")
uploaded_file = st.sidebar.file_uploader("Upload Model", type=["py"])
uploaded_json_file = st.sidebar.file_uploader("Upload JSON", type=["json"])
st.session_state['py_path'] = None
st.session_state['fn_names'] = ["feasibility_restoration",
                                "sensitivity_analysis",
                                "components_retrival",
                                "evaluate_modification",
                                "external_tools"]

interpreter, explainer, engineer, coordinator = get_agents(st.session_state.fn_names,
                                                           st.session_state.client,
                                                           st.session_state.gpt_model)
st.session_state['Interpreter'] = interpreter
st.session_state['Explainer'] = explainer
st.session_state['Engineer'] = engineer
st.session_state['Coordinator'] = coordinator


if not st.session_state.get("messages"):
    st.session_state["messages"] = []

if not st.session_state.get("team_conversation"):
    st.session_state["team_conversation"] = []

if not st.session_state.get("chat_history"):
    st.session_state["chat_history"] = []

if not st.session_state.get("detailed_chat_history"):
    st.session_state["detailed_chat_history"] = []


def process():
    if uploaded_file is None:
        st.error("Please upload your model first.")
        return

    if uploaded_json_file is None:
        st.error("Please upload your json file first.")
        return

    data = json.load(uploaded_json_file)
    # Store data in session state
    st.session_state['data'] = data
    
    models_dict, code = initial_loading(uploaded_file, data)

    with st.chat_message("user"):
        st.markdown("I have uploaded a Pyomo model.")
    st.session_state.messages.append({"role": "user", "content": "I have uploaded a Pyomo model."})
    
    # interpret the model components
    models_dict, cnt, completion = st.session_state.Interpreter.generate_interpretation_exp(st.session_state,
                                                                                            models_dict, code)
    st.session_state['models_dict'] = models_dict
    st.session_state['code'] = code
    
    # update model representation with component descriptions
    update_model_representation(st.session_state.models_dict)
    # illustrate the model
    illustration_stream = st.session_state.Interpreter.generate_illustration_exp(st.session_state,
                                                                                 models_dict["model_representation"])
    with st.chat_message("assistant"):
        illustration = st.write_stream(illustration_stream)
    # update model representation with model description
    st.session_state.models_dict['model_1']['model description'] = illustration
    update_model_representation(st.session_state.models_dict)
    # if the model is infeasible, generate inference
    if st.session_state.models_dict['model_1']['model status'] in [TerminationCondition.infeasible,
                                                                   TerminationCondition.infeasibleOrUnbounded]:
        inference_stream = st.session_state.Interpreter.generate_inference_exp(st.session_state,
                                                                               st.session_state.models_dict["model_representation"])
        with st.chat_message("assistant"):
            inference = st.write_stream(inference_stream)
        # update model representation with inference description
        st.session_state.models_dict['model_1']['model description'] = illustration + '\n' + inference
        update_model_representation(st.session_state.models_dict)

    # append model representation to messages
    st.session_state.messages.append({"role": "assistant",
                                      "content": st.session_state.models_dict["model_representation"]["model description"]})

    # append detailed chat history
    st.session_state.chat_history.append("user: I have uploaded a Pyomo model.")
    st.session_state.chat_history.append("assistant: " +
                                         st.session_state.models_dict["model_representation"]["model description"])
    st.session_state.detailed_chat_history.append("user: I have uploaded a Pyomo model.")
    st.session_state.detailed_chat_history.append("assistant: " +
                                                  st.session_state.models_dict["model_representation"]["model description"])

    # save model_description and description of every component
    if not os.path.exists("logs/model_json"):
        os.makedirs("logs/model_json")
    if not os.path.exists("logs/code_draft"):
        os.makedirs("logs/code_draft")
    if not os.path.exists("logs/ilps"):
        os.makedirs("logs/ilps")

    json2save = get_skipJSON(st.session_state.models_dict["model_representation"])
    with open(f"logs/model_json/{os.path.splitext(uploaded_file.name)[0]}.json", "w") as f:
        json.dump(json2save, f)


def load_json():
    if uploaded_file is None:
        st.error("Please upload your model first.")
        return
    if uploaded_json_file is None:
        st.error("Please upload your json file first.")
        return

    data = json.load(uploaded_json_file)
    # Store data in session state
    st.session_state['data'] = data
    
    models_dict, code = initial_loading(uploaded_file, data)

    with st.chat_message("user"):
        st.markdown("I have uploaded a Pyomo model.")
    st.session_state.messages.append({"role": "user", "content": "I have uploaded a Pyomo model."})

    skipJSON = json.load(uploaded_json_file)
    models_dict = feed_skipJSON(skipJSON, models_dict)

    st.session_state["models_dict"] = models_dict
    st.session_state['code'] = code
    # update model representation with component and model descriptions
    update_model_representation(st.session_state.models_dict)

    time.sleep(8)
    stream = string_generator(skipJSON["model description"])
    with st.chat_message("assistant"):
        st.write_stream(stream)

    # append model representation to messages
    st.session_state.messages.append({"role": "assistant",
                                      "content": st.session_state.models_dict["model_representation"][
                                          "model description"]})

    # append detailed chat history
    st.session_state.chat_history.append("user: I have uploaded a Pyomo model.")
    st.session_state.chat_history.append("assistant: " +
                                         st.session_state.models_dict["model_representation"]["model description"])
    st.session_state.detailed_chat_history.append("user: I have uploaded a Pyomo model.")
    st.session_state.detailed_chat_history.append("assistant: " +
                                                  st.session_state.models_dict["model_representation"][
                                                      "model description"])


chat_history_texts = '\n\n'.join(st.session_state.chat_history)
detailed_chat_history_texts = '\n\n'.join(st.session_state.detailed_chat_history)

st.sidebar.button("Process", on_click=process)
st.sidebar.button("Load JSON", on_click=load_json)


show_model_representation = st.sidebar.checkbox("Show Model Representation", False)
model_representation_placeholder = st.empty()
show_code = st.sidebar.checkbox("Show Code", False)
code_placeholder = st.empty()
show_tech_feedback = st.sidebar.checkbox("Show Technical Feedback", False)
tech_feedback_placeholder = st.empty()

st.sidebar.download_button(label="Export Chat History", data=chat_history_texts,
                           file_name='chat_history.txt', mime='text/plain')
st.sidebar.download_button(label="Export Detailed Chat History", data=detailed_chat_history_texts,
                           file_name='detailed_chat_history.txt', mime='text/plain')


st.sidebar.markdown("### Status")
status = st.sidebar.empty()

st.sidebar.markdown("### Round")
cur_round = st.sidebar.empty()

st.sidebar.markdown("### Agent")
agent_name = st.sidebar.empty()

st.sidebar.markdown("### Task")
task = st.sidebar.empty()


if show_model_representation:
    with model_representation_placeholder.container():
        st.json(st.session_state.models_dict["model_representation"])
else:
    model_representation_placeholder.empty()

if show_code:
    with code_placeholder.container():
        st.code(st.session_state.code)
else:
    code_placeholder.empty()

if show_tech_feedback:
    with tech_feedback_placeholder.container():
        for message in st.session_state.team_conversation:
            st.write(message['agent_name'] + ': ' + message['agent_response'])
            # if message['agent_name'] in ['Programmer', 'Operator', 'Syntax reminder', 'Explainer', 'Coordinator']:
            #     st.write(message['agent_name'] + ': ' + message['agent_response'])
else:
    tech_feedback_placeholder.empty()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "visualization" in message:
            if message["visualization"] == "heatmap":
                visualizer = SupplyChainVisualizer()
                fig = visualizer.create_facility_heatmap(st.session_state.models_dict)
                st.pyplot(fig)
            
            if "analysis" in message:
                analysis = message["analysis"]
                st.write("\n### Capacity Analysis")
                st.write(f"**CPU Capacity vs Demand:**")
                st.write(f"- Total Capacity: {analysis['cpu']['capacity']:,.0f} units")
                st.write(f"- Total Demand: {analysis['cpu']['demand']:,.0f} units")
                if analysis['cpu']['capacity'] > 0:
                    st.write(f"- Utilization: {analysis['cpu']['utilization']:.1f}%")
                else:
                    st.write("- Utilization: N/A (No capacity available)")
                
                st.write(f"\n**GPU Capacity vs Demand:**")
                st.write(f"- Total Capacity: {analysis['gpu']['capacity']:,.0f} units")
                st.write(f"- Total Demand: {analysis['gpu']['demand']:,.0f} units")
                if analysis['gpu']['capacity'] > 0:
                    st.write(f"- Utilization: {analysis['gpu']['utilization']:.1f}%")
                else:
                    st.write("- Utilization: N/A (No capacity available)")

def handle_user_input(prompt):
    # Check if data and model are loaded
    if 'data' not in st.session_state:
        st.error("Please upload your model and data files first.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create or update models_dict structure for visualization
    if 'models_dict' not in st.session_state:
        st.session_state.models_dict = {}
    
    # Convert string tuple keys to actual tuples for all data
    demand_data = {}
    fab_capacity_data = {}
    trans_cost_data = {}
    revenue_data = {}
    
    # Process Demand data
    for key, value in st.session_state['data']['Demand'].items():
        if isinstance(key, tuple):
            tuple_key = key
        else:
            try:
                tuple_key = ast.literal_eval(key)
            except:
                clean_key = key.strip("()' ").split("', '")
                tuple_key = tuple(clean_key)
        demand_data[tuple_key] = value

    # Process Revenue data
    for key, value in st.session_state['data']['Revenue'].items():
        if isinstance(key, tuple):
            tuple_key = key
        else:
            try:
                tuple_key = ast.literal_eval(key)
            except:
                clean_key = key.strip("()' ").split("', '")
                tuple_key = tuple(clean_key)
        revenue_data[tuple_key] = value

    # Process FabCapacity data
    for key, value in st.session_state['data']['FabCapacity'].items():
        if isinstance(key, tuple):
            tuple_key = key
        else:
            try:
                tuple_key = ast.literal_eval(key)
            except:
                clean_key = key.strip("()' ").split("', '")
                tuple_key = tuple(clean_key)
        fab_capacity_data[tuple_key] = value

    # Process TransCost data
    for key, value in st.session_state['data']['TransCost'].items():
        if isinstance(key, tuple):
            tuple_key = key
        else:
            try:
                tuple_key = ast.literal_eval(key)
            except:
                clean_key = key.strip("()' ").split("', '")
                tuple_key = tuple(clean_key)
        trans_cost_data[tuple_key] = value

    # Create the models_dict with proper data structure
    st.session_state.models_dict['model_1'] = {
        'Sets': {
            'Stores': st.session_state['data']['Stores'],
            'Products': st.session_state['data']['Products'],
            'Fabs': [f for f in st.session_state['data']['Fabs'] if f.startswith('fab')],
            'Assemblies': st.session_state['data']['Assemblies']
        },
        'Parameters': {
            'Demand': demand_data,
            'Revenue': revenue_data,
            'FabCapacity': fab_capacity_data,
            'TransCost': trans_cost_data
        }
    }

    # Define visualization keywords and their corresponding functions
    visualization_types = {
        "heatmap": ["heatmap", "capacity distribution", "facility capacity"],
        "demand": ["demand distribution", "demand pattern", "show demand"],
        "production": ["production allocation", "production distribution"],
        "transportation": ["transportation cost", "shipping cost", "logistics cost"],
        "sunburst": ["sunburst", "utilization breakdown", "capacity utilization"],
        "flow": ["supply chain flow", "material flow", "sankey"]
    }

    # Check for visualization requests
    prompt_lower = prompt.lower()
    visualization_requested = False
    
    for viz_type, keywords in visualization_types.items():
        if any(keyword in prompt_lower for keyword in keywords):
            visualization_requested = True
            response = handle_visualization_request(viz_type, st.session_state.models_dict)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            break

    # If not a visualization request, check for analysis requests
    if not visualization_requested:
        analysis_keywords = [
            "profitability", "profit", "revenue", "store performance",
            "price", "pricing", "optimization",
            "transportation", "shipping", "logistics",
            "production", "allocation", "capacity"
        ]

        if any(keyword in prompt_lower for keyword in analysis_keywords):
            response = handle_analysis_request(prompt, st.session_state.models_dict)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Handle other types of requests using existing workflow
            updated_messages, team_conversation = OptiChat_workflow_exp(
                st.session_state,
                st.session_state.Coordinator,
                st.session_state.Engineer,
                st.session_state.Explainer,
                st.session_state.messages,
                st.session_state.models_dict
            )
            st.session_state.messages = updated_messages

# Add analysis buttons to sidebar
st.sidebar.markdown("### Supply Chain Analysis")
if st.sidebar.button("Analyze Store Profitability"):
    handle_user_input("Show store profitability analysis")
if st.sidebar.button("Optimize Pricing"):
    handle_user_input("Analyze pricing opportunities")
if st.sidebar.button("Optimize Transportation"):
    handle_user_input("Analyze transportation costs")
if st.sidebar.button("Optimize Production"):
    handle_user_input("Show production allocation analysis")

# Accept user input
if prompt := st.chat_input("Enter your query here..."):
    handle_user_input(prompt)


def tuple_keys(d):
    return {ast.literal_eval(k): v for k, v in d.items()}

def create_model(data):
    model = ConcreteModel()

    # Sets
    model.Products = Set(initialize=data['Products'])
    model.Stores = Set(initialize=data['Stores'])
    model.Fabs = Set(initialize=data['Fabs'])
    model.Assemblies = Set(initialize=data['Assemblies'])

    # Convert string keys to tuples for all relevant parameters
    demand = tuple_keys(data['Demand'])
    revenue = tuple_keys(data['Revenue'])
    fab_capacity = tuple_keys(data['FabCapacity'])
    fab_cost = tuple_keys(data['FabCost'])
    trans_cost = tuple_keys(data['TransCost'])
    trans_capacity = tuple_keys(data['TransCapacity'])

    # Parameters
    model.Demand = Param(model.Stores, model.Products, initialize=demand, default=0)
    model.Revenue = Param(model.Stores, model.Products, initialize=revenue, default=0)
    model.FabCapacity = Param(model.Fabs, model.Products, initialize=fab_capacity, default=0)
    model.FabCost = Param(model.Fabs, model.Products, initialize=fab_cost, default=0)
    model.TransCost = Param(model.Fabs, model.Assemblies, model.Products, initialize=trans_cost, default=0)
    model.TransCapacity = Param(model.Fabs, model.Assemblies, model.Products, initialize=trans_capacity, default=0)

    print("Stores:", data['Stores'])
    print("Products:", data['Products'])
    print("Demand keys:", list(tuple_keys(data['Demand']).keys()))

    # ... rest of your code ...
    return model

def plot_demand_distribution(model):
    stores = list(model.Stores)
    products = list(model.Products)
    
    # Collect data
    demand_data = {s: [value(model.Demand[s, p]) for p in products] for s in stores}
    
    # Plot
    fig, ax = plt.subplots()
    for s, demands in demand_data.items():
        ax.plot(products, demands, marker='o', label=s)
    
    ax.set_xlabel('Products')
    ax.set_ylabel('Demand')
    ax.set_title('Demand Distribution by Store')
    ax.legend()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

def handle_visualization(request):
    if 'data' not in st.session_state:
        st.error("Please upload your model and data first.")
        return
    
    # Create a models_dict structure as expected by the visualization functions
    models_dict = {
        'model_1': {
            'Sets': {
                'Stores': st.session_state['data']['Stores'],
                'Products': st.session_state['data']['Products'],
                'Fabs': [f for f in st.session_state['data']['Fabs'] if f.startswith('fab')]
            },
            'Parameters': {
                'Demand': st.session_state['data']['Demand']
            }
        }
    }
    
    response = handle_visualization_request(request, models_dict)
    st.write(response)

# Add visualization buttons
st.sidebar.markdown("### Visualizations")
if st.sidebar.button("Show Demand Distribution"):
    handle_visualization("show demand patterns")
if st.sidebar.button("Show Production by Fab"):
    handle_visualization("show production by fab")

def create_capacity_heatmap():
    # Filter only manufacturing facilities (fab1-fab6)
    fabs = [f for f in st.session_state['data']['Fabs'] if f.startswith('fab')]
    products = st.session_state['data']['Products']

    # Create capacity matrix
    capacity_matrix = np.zeros((len(fabs), len(products)))
    for i, fab in enumerate(fabs):
        for j, product in enumerate(products):
            # Try different key formats since JSON might store tuples differently
            possible_keys = [
                f"('{fab}', '{product}')",  # Original format
                f"({fab}, {product})",      # Without quotes
                str((fab, product)),        # Python tuple string
                f"{fab},{product}"          # Simple comma format
            ]
            
            # Try to find the capacity value using different key formats
            capacity_value = 0
            for key in possible_keys:
                if key in st.session_state['data']['FabCapacity']:
                    capacity_value = float(st.session_state['data']['FabCapacity'][key])
                    break
                # Also check if the key exists as a tuple
                elif (fab, product) in st.session_state['data']['FabCapacity']:
                    capacity_value = float(st.session_state['data']['FabCapacity'][(fab, product)])
                    break
            
            capacity_matrix[i, j] = capacity_value
    
    # Debug information
    st.write("### Debug Information")
    st.write("FabCapacity Data Structure:")
    st.write(st.session_state['data']['FabCapacity'])
    
    # Create and display heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = sns.heatmap(capacity_matrix, 
                         annot=True, 
                         fmt='.0f', 
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
    total_cpu_capacity = sum(float(st.session_state['data']['FabCapacity'].get(f"('fab{i}', 'cpu')", 0)) for i in range(1, 7))
    total_gpu_capacity = sum(float(st.session_state['data']['FabCapacity'].get(f"('fab{i}', 'gpu')", 0)) for i in range(1, 7))
    total_cpu_demand = sum(float(st.session_state['data']['Demand'].get(f"('store{i}', 'cpu')", 0)) for i in range(1, 7))
    total_gpu_demand = sum(float(st.session_state['data']['Demand'].get(f"('store{i}', 'gpu')", 0)) for i in range(1, 7))

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

    return fig

def create_capacity_analysis():
    total_cpu_capacity = sum(float(st.session_state['data']['FabCapacity'].get(f"('fab{i}', 'cpu')", 0)) for i in range(1, 7))
    total_gpu_capacity = sum(float(st.session_state['data']['FabCapacity'].get(f"('fab{i}', 'gpu')", 0)) for i in range(1, 7))
    total_cpu_demand = sum(float(st.session_state['data']['Demand'].get(f"('store{i}', 'cpu')", 0)) for i in range(1, 7))
    total_gpu_demand = sum(float(st.session_state['data']['Demand'].get(f"('store{i}', 'gpu')", 0)) for i in range(1, 7))

    analysis = {
        'cpu': {
            'capacity': total_cpu_capacity,
            'demand': total_cpu_demand,
            'utilization': (total_cpu_demand/total_cpu_capacity*100) if total_cpu_capacity > 0 else 0
        },
        'gpu': {
            'capacity': total_gpu_capacity,
            'demand': total_gpu_demand,
            'utilization': (total_gpu_demand/total_gpu_capacity*100) if total_gpu_capacity > 0 else 0
        }
    }
    return analysis
