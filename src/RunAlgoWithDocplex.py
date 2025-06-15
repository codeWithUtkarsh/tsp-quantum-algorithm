import pickle
import time
import yaml
import gc
import networkx as nx
from matplotlib import pyplot as plt

# from Exterpolate import extrapolate
from qiskit_utils.converters import QuadraticProgramToQubo
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_utils.applications import Tsp
from qiskit_utils.algorithms import MinimumEigenOptimizer


class QuantumTSPRunner:
    def __init__(self, config_file="config.yaml"):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.available_optimizers = config['optimizers']
        self.number_of_nodes_quantum = config['number_of_nodes_quantum']
        self.result_directory = config['result_directory']
        self.results = {}
        self.output_dir = config.get('output_dir', '.')  # Default to current directory

    def run_quantum_qaoa(self, qubo, edge_prob=0.5):
        """Run QAOA algorithm for TSP"""
        global qaoa_sample_measure_time
        qaoa_time_taken = []
        qaoa_sample_measure_time = []
        qaoa_eval_count = []
        iteration_count = []

        def store_intermediate_result(eval_count, parameters, evaluated_value, metadata):
            qaoa_time_taken.append(metadata['simulator_metadata']['time_taken'])
            qaoa_sample_measure_time.append(metadata['simulator_metadata']['sample_measure_time'])
            qaoa_eval_count.append(eval_count)
            iteration_count.append(1)

        sampler = Sampler(
            backend_options={"method": "automatic"},
            run_options={"shots": 1024, "seed": 42}
        )

        opt = COBYLA(maxiter=150, rhobeg=3.14159, tol=1e-4)
        qaoa = QAOA(
            optimizer=opt,
            sampler=sampler,
            reps=1,
            callback=store_intermediate_result
        )

        qaoa_optimizer = MinimumEigenOptimizer(qaoa)
        start_time = time.time()  # Seconds
        print("Start time ::", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
        result = qaoa_optimizer.solve(qubo)
        end_time = time.time()  # Seconds
        function_execution_time = end_time - start_time
        print("End time ::", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
        print(f"Execution took :: {function_execution_time:.2f} seconds")

        del sampler
        del qaoa
        del qaoa_optimizer
        del opt
        gc.collect()

        return qaoa_time_taken, qaoa_sample_measure_time, qaoa_eval_count, len(iteration_count), result

    def process_sample(self, sample_name, num_vertices, qubo):
        """Process a single sample and return its results"""
        print(f"\n{sample_name}")
        print("-" * len(sample_name))

        sample_results = {}

        quantum_runtime_stats = {}
        qaoa_time_taken, qaoa_sample_measure_time, qaoa_eval_count, iteration_count, solution_result = self.run_quantum_qaoa(
            qubo
        )
        res_def = {
            "qaoa_time": qaoa_time_taken,
            "qaoa_sample_measure_time": qaoa_sample_measure_time,
            "qaoa_eval_value": qaoa_eval_count,
            "iteration_count": iteration_count,
        }

        quantum_runtime_stats[num_vertices] = res_def
        return quantum_runtime_stats, solution_result

    def draw_tsp_solution(self, G, order, colors, pos):
        G2 = nx.DiGraph()
        G2.add_nodes_from(G)
        n = len(order)
        for i in range(n):
            j = (i + 1) % n
            G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(
            G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
        )
        edge_labels = nx.get_edge_attributes(G2, "weight")
        nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)

    def save_results(self, filename):
        """Save current results to a Pickle file"""
        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'wb') as pickle_file:
            pickle.dump(self.results, pickle_file)

        print(f"Dictionary has been saved to {filepath} successfully.")
        return filepath

    def run_all_samples(self):
        total_samples = len(self.number_of_nodes_quantum)

        for idx, current_node_count in enumerate(self.number_of_nodes_quantum):
            tsp = Tsp.create_random_instance(current_node_count, seed=123)
            adj_matrix = nx.to_numpy_array(tsp.graph)
            quadratic_problem = tsp.to_quadratic_program()
            quadratic_problem_to_qubo = QuadraticProgramToQubo()
            qubo = quadratic_problem_to_qubo.convert(quadratic_problem)
            qubit_op, offset = qubo.to_ising()

            (sample_name, adj_matrix) = (f'Node {current_node_count}', adj_matrix)

            print(f"Processing sample {idx + 1}/{total_samples}: {sample_name}")

            sample_results, solution_result = self.process_sample(sample_name, current_node_count, qubo)

            # print("energy:", solution_result.min_eigen_solver_result.eigenvalue)
            # print("tsp objective:", solution_result.min_eigen_solver_result.eigenvalue + offset)
            # x = tsp.sample_most_likely(solution_result.min_eigen_solver_result.eigenstate)
            # print("feasible:", qubo.is_feasible(x))
            # z = tsp.interpret(x)
            # print("solution:", z)
            # print("solution objective:", tsp.tsp_value(z, adj_matrix))
            # colors = ["r" for node in tsp.graph.nodes]
            # pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]
            # self.draw_tsp_solution(tsp.graph, z, colors, pos)

            if 'COBYLA' in self.results.keys():
                self.results['COBYLA'].update(sample_results)
            else:
                self.results['COBYLA'] = sample_results

            del tsp
            del adj_matrix
            del quadratic_problem
            del quadratic_problem_to_qubo
            del qubo
            del qubit_op
            del offset
            del sample_results
            gc.collect()

            print(f"Completed sample: {sample_name}. Intermediate results saved.")

        self.save_results(f"{self.result_directory}/algo_with_docplex/quantum_tsp_results_final.pkl")
        print("All processing complete.")

        return self.results

