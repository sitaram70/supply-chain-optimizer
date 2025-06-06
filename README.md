# Supply Chain Optimizer

A Python-based optimization tool for supply chain management that helps businesses make data-driven decisions about production, transportation, and inventory management.

## Features

- Multi-facility production planning
- Transportation optimization
- Demand fulfillment optimization
- Profit maximization
- Capacity constraint handling
- JSON-based data input support

## Prerequisites

- Python 3.8+
- Gurobi Optimizer
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/supply-chain-optimizer.git
cd supply-chain-optimizer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Ensure you have a valid Gurobi license installed.

## Project Structure

```
supply-chain-optimizer/
├── data/                  # Data files (JSON, CSV)
├── Feas/                  # Feasible model implementations
├── Infeas/               # Infeasible model implementations
├── logs/                 # Log files
├── agents.py             # Agent implementations
├── app.py               # Main application file
├── utils.py             # Utility functions
└── requirements.txt     # Project dependencies
```

## Usage

1. Prepare your supply chain data in CSV format
2. Convert the data to JSON format using the provided tools
3. Run the optimization model:
```bash
python app.py
```

## Data Format

The system expects supply chain data in JSON format with the following structure:
- Demand data
- Processing capabilities
- Transportation costs and capacities
- Facility information

See the example files in the `data/` directory for reference.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
#   s u p p l y - c h a i n - o p t i m i z e r  
 