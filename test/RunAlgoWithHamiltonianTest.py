import time
import yaml
import pickle
import gc
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import QAOA


from src.TSPHamiltonian import TSPHamiltonian
from test.Node5BruteforceForTravellingSalesman import Node5BruteforceForTravellingSalesman


class QuantumTSPRunner:
    """Class to run quantum TSP experiments and manage resources efficiently"""

    def __init__(self, config_file="../config.yaml"):
        """Initialize with configuration from YAML file"""
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.available_optimizers = config['optimizers']
        self.number_of_nodes_quantum = config['number_of_nodes_quantum']
        self.result_directory = config['result_directory']
        self.results = {}
        self.output_dir = config.get('output_dir', '.')  # Default to current directory

    def run_quantum_qaoa(self, num_vertices, adj_matrix, optimizer, edge_prob=0.5):
        """Run QAOA algorithm for TSP"""
        qaoa_eval_value = []
        qaoa_intermediate_parameters = []
        qaoa_time = []
        qaoa_simulation_time = []
        qaoa_count = []
        metadata_ret = []
        iteration_count = []

        def store_intermediate_result(eval_count, parameters, evaluated_value, metadata):
            # qaoa_eval_value.append(evaluated_value)
            # qaoa_intermediate_parameters.append(parameters)
            qaoa_time.append(metadata['simulator_metadata']['time_taken'])
            qaoa_simulation_time.append(metadata['simulator_metadata']['sample_measure_time'])
            qaoa_count.append(eval_count)  # Amount of evaluation on quantum made
            iteration_count.append(1)
            # metadata_ret.append(metadata)

        if adj_matrix is None:
            tsp = TSPHamiltonian(num_vertices)
            adj_matrix = tsp._generate_random_distances()

        hamiltonian, tsp = TSPHamiltonian.create_tsp_from_adjacency_matrix(adj_matrix)

        sampler = Sampler(
            backend_options={"method": "automatic"},
            run_options={"shots": 1024, "seed": 42}
        )

        if optimizer == 'COBYLA':
            opt = COBYLA(maxiter=150, rhobeg=3.14159, tol=1e-4)  # rhobeg set to π as suggested in the paper
        else:
            raise ValueError(f"Optimizer '{optimizer}' not recognized.")

        qaoa = QAOA(
            optimizer=opt,
            sampler=sampler,
            reps=1,
            callback=store_intermediate_result
        )

        start_time = time.time()  # Seconds
        print("Start time ::", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        end_time = time.time()  # Seconds
        function_execution_time = end_time - start_time
        print("End time ::", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
        print(f"Execution took :: {function_execution_time:.2f} seconds")

        # Free memory
        del hamiltonian
        del tsp
        del sampler
        del qaoa
        del result
        gc.collect()

        return qaoa_time, qaoa_eval_value, len(iteration_count)

    @staticmethod
    def convert_to_serializable(obj):
        """Convert objects to JSON serializable format"""
        if hasattr(obj, 'tolist'):  # For numpy arrays
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: QuantumTSPRunner.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [QuantumTSPRunner.convert_to_serializable(i) for i in obj]
        else:
            return obj

    def process_sample(self, sample_name, adj_matrix):
        """Process a single sample and return its results"""
        print(f"\n{sample_name}")
        print("-" * len(sample_name))

        num_vertices = len(adj_matrix)
        sample_results = {}

        for optimizer in self.available_optimizers:
            quantum_runtime_stats = {}
            qaoa_time, qaoa_eval_value, iteration_count = self.run_quantum_qaoa(
                num_vertices, adj_matrix, optimizer
            )
            res_def = {
                "time": qaoa_time,
                # "evaluated_value": qaoa_eval_value,
                "iteration": iteration_count,
            }

            quantum_runtime_stats[num_vertices] = res_def

            sample_results[f"{sample_name}_{optimizer}"] = quantum_runtime_stats
            print(f'\rOptimization complete for optimizer: {optimizer}')

        return sample_results

    def save_results(self, filename):
        """Save current results to a Pickle file"""
        filepath = f".{self.output_dir}/{filename}"
        with open(filepath, 'wb') as pickle_file:
            pickle.dump(self.results, pickle_file)

        print(f"Dictionary has been saved to {filepath} successfully.")
        return filepath

    def run_all_samples(self):
        """Run all samples from Node5BruteforceForTravellingSalesman"""
        tsp_solver = Node5BruteforceForTravellingSalesman()
        total_samples = len(tsp_solver.distance_matrices)

        for idx, (sample_name, adj_matrix) in enumerate(tsp_solver.distance_matrices):
            print(f"Processing sample {idx + 1}/{total_samples}: {sample_name}")

            sample_results = self.process_sample(sample_name, adj_matrix)

            self.results.update(sample_results)
            self.save_results(f"{self.result_directory}/algo_with_hamiltonian/quantum_tsp_results_intermediate_{idx + 1}.pickle")

            gc.collect()
            print(f"Completed sample: {sample_name}. Intermediate results saved.")

        self.save_results(f"{self.result_directory}/algo_with_hamiltonian/quantum_tsp_results.pickle")
        print("All processing complete.")

        return self.results

runner = QuantumTSPRunner()
runner.run_all_samples()