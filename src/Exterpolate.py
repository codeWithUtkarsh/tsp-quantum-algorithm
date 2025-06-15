import numpy as np
import yaml
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit


config_file = "config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
number_of_nodes_quantum = config['number_of_nodes_quantum']
output_dir = config.get('output_dir', '.saved_result')
result_directory = config['result_directory']

def read_classical_data(file_path):
    number_of_nodes_classical = []
    execution_times = []
    try:
        with open(file_path, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        nodes = int(parts[0].strip())
                        time = float(parts[1].strip())
                        number_of_nodes_classical.append(nodes)
                        execution_times.append(time)
                    except ValueError:
                        print(f"Warning: Could not parse line: {line}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")

    _x_classical = np.array(number_of_nodes_classical)
    _y_classical = np.array(execution_times) / 1e9  # Convert nanoseconds to seconds

    return _x_classical, _y_classical

def read_quantum_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            loaded_dict = pickle.load(f)



        time_quantum_dict = loaded_dict
        _x_quantum = np.array(number_of_nodes_quantum)

        _y_quantum = np.array([
            np.array(time_quantum_dict['COBYLA'][3]['qaoa_sample_measure_time']).mean(),
            np.array(time_quantum_dict['COBYLA'][4]['qaoa_sample_measure_time']).mean(),
            np.array(time_quantum_dict['COBYLA'][5]['qaoa_sample_measure_time']).mean()])
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(f"Error loading quantum time data") from e

    return _x_quantum, _y_quantum

def extrapolate(classical_time_file_path, quantum_time_file_path, use_log_scale=False):

    x_classical, y_classical = read_classical_data(classical_time_file_path)
    x_quantum, y_quantum = read_quantum_data(quantum_time_file_path)

    def quantum_complexity_model(N, a, b):
        return a * (N ** 4) + b

    # Prevent negative coefficients and extreme values
    bounds = ([1e-10, -1], [1e-3, 1])

    try:
        curve_fitting_coefficient = curve_fit(
            quantum_complexity_model, x_quantum, y_quantum,
            bounds=bounds, maxfev=5000)
        popt_quantum = curve_fitting_coefficient[0]
        print("Quantum complexity parameters (a, b):", popt_quantum) # popt_quantum has optimum coefficient
    except RuntimeError:
        print("Error fitting quantum model, using fallback values")
        exit(1)

    # Create a smooth range for plotting
    x_smooth = np.linspace(min(min(x_classical), min(x_quantum)),
                           25, 1000)

    y_quantum_smooth = quantum_complexity_model(x_smooth, *popt_quantum)

    # Plotting
    plt.figure(figsize=(12, 7))

    # Plot raw data points
    plt.scatter(x_classical, y_classical, color='blue', s=60, label='Classical Data')
    plt.scatter(x_quantum, y_quantum, color='red', s=60, label='Quantum Data')

    # Connect classical data points with lines
    plt.plot(x_classical, y_classical, 'b-', alpha=0.6, linewidth=1.5, label='Classical (Actual Values)')

    # Quantum model - split into observed and extrapolated
    # Define the ranges
    quantum_range_indices = (x_smooth >= min(x_quantum)) & (x_smooth <= max(x_quantum))
    extrapolation_indices = x_smooth > max(x_quantum)

    # Observed range
    plt.plot(x_smooth[quantum_range_indices],
             y_quantum_smooth[quantum_range_indices],
             color='red', linewidth=2,
             label='Quantum Model: O(n⁴)')

    # Extrapolated range
    plt.plot(x_smooth[extrapolation_indices],
             y_quantum_smooth[extrapolation_indices],
             color='red', linestyle='--', linewidth=2,
             label='Quantum Model (Extrapolated)')

    # Use log scale for y-axis
    if use_log_scale:
        plt.yscale('log')
        scale = 'log scale'
    else:
        plt.yscale('linear')
        scale = 'in sec'

    # Add informative title and labels
    plt.title("Classical vs Quantum Computing Complexity for TSP", fontsize=16)
    plt.xlabel("Number of Nodes (N)", fontsize=14)
    plt.ylabel(f"Computation Time ({scale})", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.figtext(0.5, 0.01,
                f"Classical: Simulated Annealing | Quantum complexity: QAOA (O(n⁴))",
                ha="center", fontsize=12,
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    # Add legend
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return plt


def extrapolate_quantum(quantum_time_file_path, use_log_scale=False):

    x_quantum, y_quantum = read_quantum_data(quantum_time_file_path)
    def quantum_complexity_model(N, a, b):
        return a * (N ** 4) + b

    bounds = ([1e-10, -1], [1e-3, 1])

    try:
        curve_fitting_coefficient = curve_fit(
            quantum_complexity_model, x_quantum, y_quantum,
            bounds=bounds, maxfev=5000)
        popt_quantum = curve_fitting_coefficient[0]
        print("Quantum complexity parameters (a, b):", popt_quantum) # popt_quantum has optimum coefficient
    except RuntimeError:
        print("Error fitting quantum model, using fallback values")
        exit(1)

    # Create a smooth range for plotting
    x_smooth = np.linspace(min(x_quantum),
                           25, 1000)
    y_quantum_smooth = quantum_complexity_model(x_smooth, *popt_quantum)

    # Plotting
    plt.figure(figsize=(12, 7))

    # Plot raw data points
    plt.scatter(x_quantum, y_quantum, color='red', s=60, label='Quantum Data')

    # Quantum model - split into observed and extrapolated
    # Define the ranges
    quantum_range_indices = (x_smooth >= min(x_quantum)) & (x_smooth <= max(x_quantum))
    extrapolation_indices = x_smooth > max(x_quantum)

    # Observed range
    plt.plot(x_smooth[quantum_range_indices],
             y_quantum_smooth[quantum_range_indices],
             color='red', linewidth=2,
             label='Quantum Model: O(n⁴)')

    # Extrapolated range
    plt.plot(x_smooth[extrapolation_indices],
             y_quantum_smooth[extrapolation_indices],
             color='red', linestyle='--', linewidth=2,
             label='Quantum Model (Extrapolated)')

    # Use log scale for y-axis
    if use_log_scale:
        plt.yscale('log')
        scale = 'log scale'
    else:
        plt.yscale('linear')
        scale = 'in sec'

    # Add informative title and labels
    plt.title("Classical vs Quantum Computing Complexity for TSP", fontsize=16)
    plt.xlabel("Number of Nodes (N)", fontsize=14)
    plt.ylabel(f"Computation Time ({scale})", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    # Customize tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.figtext(0.5, 0.01,
                f"Classical: Simulated Annealing | Quantum complexity: QAOA (O(n⁴))",
                ha="center", fontsize=12,
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    # Add legend
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return plt
