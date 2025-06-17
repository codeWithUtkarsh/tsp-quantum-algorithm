# Quantum TSP Solver with QAOA

## Overview

This project implements a quantum-enhanced Traveling Salesman Problem (TSP) solver using the Quantum Approximate Optimization Algorithm (QAOA). The implementation leverages Qiskit's quantum computing framework to solve TSP instances and compare quantum vs classical optimization approaches.

## Algorithm Description

### QAOA-based TSP Algorithm

The core algorithm implements a quantum approach to solving the TSP using the following steps:

#### 1. TSP Problem Formulation
- **Random Instance Generation**: Creates TSP instances with specified number of nodes using NetworkX
- **Graph Representation**: Generates complete graphs with random edge weights
- **Adjacency Matrix**: Converts graph to numpy adjacency matrix for optimization

#### 2. Quantum Optimization Pipeline

```python
# Algorithm Flow:
1. TSP Instance Creation
   └── Tsp.create_random_instance(n_nodes, seed=123)

2. Quadratic Program Conversion
   └── tsp.to_quadratic_program()

3. QUBO Transformation
   └── QuadraticProgramToQubo().convert(quadratic_program)

4. Ising Model Conversion
   └── qubo.to_ising() → (qubit_op, offset)

5. QAOA Optimization
   └── MinimumEigenOptimizer(qaoa).solve(qubo)
```

#### 3. QAOA Configuration

**Quantum Components:**
- **Sampler**: Qiskit Aer Sampler with 1024 shots
- **Backend**: Automatic method selection
- **Seed**: Fixed at 42 for reproducibility

**Classical Optimizer:**
- **Algorithm**: COBYLA (Constrained Optimization BY Linear Approximation)
- **Max Iterations**: 150
- **Initial Step Size (rhobeg)**: π (3.14159)
- **Tolerance**: 1e-4

**QAOA Parameters:**
- **Repetitions (p)**: 1 layer
- **Ansatz**: Standard QAOA ansatz with mixing and cost Hamiltonians

#### 4. Performance Metrics Tracking

The algorithm tracks comprehensive performance metrics:

```python
Metrics Collected:
├── qaoa_time_taken          # Time per QAOA iteration
├── qaoa_sample_measure_time # Quantum sampling time
├── qaoa_eval_count         # Function evaluations
└── iteration_count         # Total iterations
```

#### 5. Solution Verification

**Energy Calculation:**
- Computes eigenvalue from quantum solution
- Adds offset to get TSP objective value
- Verifies feasibility using QUBO constraints

**Solution Interpretation:**
- Extracts most likely bitstring from quantum state
- Converts to tour representation
- Calculates actual tour distance

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/codeWithUtkarsh/tsp-quantum-algorithm.git
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `qiskit==1.1.0` - Quantum computing framework
- `qiskit-aer==0.14.2` - Quantum simulator
- `qiskit-optimization` - Optimization algorithms
- `PyYAML` - Configuration file parsing
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `networkx` - Graph operations
- `pandas` - Data manipulation

## Configuration

### config.yaml Settings

```yaml
optimizers:
  - 'COBYLA'                    # Classical optimizer for QAOA

number_of_nodes_quantum:        # TSP instance sizes to test
  - 3
  - 4  
  - 5

number_of_nodes_classical:      # For classical comparison
  - 3
  - 4
  - 5

result_directory: ".saved_result"  # Output directory for results
```

**Configuration Options:**
- **optimizers**: List of classical optimizers (currently supports COBYLA)
- **number_of_nodes_quantum**: Node counts for quantum TSP instances
- **number_of_nodes_classical**: Node counts for classical comparison
- **result_directory**: Directory for saving results and plots

## Running the Application

### Basic Execution

```bash
python main.py
```

### Execution Flow

1. **Initialization**: Loads configuration from `config.yaml`
2. **Quantum Processing**: 
   - Creates TSP instances for each configured node count
   - Runs QAOA optimization for each instance
   - Collects performance metrics
3. **Result Storage**: Saves results to pickle file
4. **Visualization**: Generates comparison plots
5. **Cleanup**: Performs memory cleanup between runs

### Expected Output

```
Processing sample 1/3: Node 3
--------------------------
Start time :: 2024-01-01 10:00:00
End time :: 2024-01-01 10:00:05
Execution took :: 5.23 seconds
Completed sample: Node 3. Intermediate results saved.

Processing sample 2/3: Node 4
--------------------------
...

All processing complete.
Dictionary has been saved to .saved_result/algo_with_docplex/quantum_tsp_results_final.pkl successfully.
Process completed successfully!
```

## Output Files

### Results Directory Structure

```
.saved_result/
└── algo_with_docplex/
    ├── quantum_tsp_results_final.pkl    # Quantum algorithm results
    └── quantum_classical_comparison_improved.png  # Comparison plot
```

### Result Data Format

The pickle file contains a nested dictionary structure:

```python
{
    'COBYLA': {
        3: {  # Number of nodes
            'qaoa_time': [...],              # Time per iteration
            'qaoa_sample_measure_time': [...], # Sampling time
            'qaoa_eval_value': [...],        # Function evaluations
            'iteration_count': int           # Total iterations
        },
        4: { ... },
        5: { ... }
    }
}
```

## Algorithm Performance

### Time Complexity
- **Problem Size**: O(n²) qubits for n-city TSP
- **QAOA Depth**: O(p) where p is the number of QAOA layers
- **Classical Optimization**: Depends on COBYLA convergence

### Space Complexity
- **Quantum State**: 2^(n²) amplitudes
- **Classical Memory**: O(n²) for adjacency matrices
- **Result Storage**: O(iterations × metrics)

### Scalability Considerations
- **Quantum Advantage**: Expected for larger instances (n > 10)
- **Classical Simulation**: Limited by exponential quantum state size
- **Hybrid Approach**: QAOA combines quantum and classical optimization

## Visualization Features

The application generates performance comparison plots showing:
- Execution time vs problem size
- Convergence behavior
- Quantum vs classical performance metrics
- Solution quality over iterations

## Troubleshooting

### Common Issues

1. **Memory Errors**: 
   - Reduce number of nodes in config.yaml
   - Ensure sufficient RAM for quantum simulation

2. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility

3. **Configuration Errors**:
   - Validate YAML syntax in config.yaml
   - Ensure result directories exist

### Debug Mode

To enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Research Applications

This implementation is suitable for:
- **Quantum Algorithm Research**: Studying QAOA performance on combinatorial problems
- **Benchmarking**: Comparing quantum vs classical optimization
- **Educational Purposes**: Understanding quantum optimization principles
- **Hybrid Algorithm Development**: Exploring quantum-classical combinations

## References

1. Farhi, E., et al. "A Quantum Approximate Optimization Algorithm" (arXiv:1411.4028)
2. Qiskit Optimization Documentation
3. COBYLA Optimization Algorithm
4. Traveling Salesman Problem Formulations

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]