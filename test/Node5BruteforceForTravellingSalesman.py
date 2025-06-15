import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations


class Node5BruteforceForTravellingSalesman:
    """
    A class to solve the Traveling Salesman Problem (TSP) for 5-node problems using brute force.
    """

    def __init__(self):
        """Initialize the class with predefined distance matrices for 5 sample problems."""
        self.distance_matrices = [
            ("Sample 1", np.array([
                [0, 12, 10, 19, 8],
                [12, 0, 3, 7, 11],
                [10, 3, 0, 5, 6],
                [19, 7, 5, 0, 9],
                [8, 11, 6, 9, 0]
            ])),
            ("Sample 2", np.array([
                [0, 4, 2, 7, 9],
                [4, 0, 6, 5, 3],
                [2, 6, 0, 8, 4],
                [7, 5, 8, 0, 11],
                [9, 3, 4, 11, 0]
            ])),
            ("Sample 3", np.array([
                [0, 15, 20, 12, 6],
                [15, 0, 9, 14, 18],
                [20, 9, 0, 7, 13],
                [12, 14, 7, 0, 10],
                [6, 18, 13, 10, 0]
            ])),
            ("Sample 4", np.array([
                [0, 3, 8, 11, 5],
                [3, 0, 6, 4, 12],
                [8, 6, 0, 7, 9],
                [11, 4, 7, 0, 2],
                [5, 12, 9, 2, 0]
            ])),
            ("Sample 5", np.array([
                [0, 18, 9, 5, 14],
                [18, 0, 13, 22, 7],
                [9, 13, 0, 16, 11],
                [5, 22, 16, 0, 8],
                [14, 7, 11, 8, 0]
            ]))
        ]

        # Store the optimal tours for each sample
        self.optimal_tours = {
            "Sample 1": "A → E → C → B → D → A",
            "Sample 2": "A → C → E → B → D → A",
            "Sample 3": "A → E → D → C → B → A",
            "Sample 4": "A → B → D → E → C → A",
            "Sample 5": "A → D → E → B → C → A"
        }

        # Store the optimal distances for each sample
        self.optimal_distances = {
            "Sample 1": 43,
            "Sample 2": 21,
            "Sample 3": 47,
            "Sample 4": 26,
            "Sample 5": 42
        }

    def brute_force_tsp(self, w, N):
        """
        Solve the TSP using brute force approach.

        Args:
            w: Distance matrix
            N: Number of nodes

        Returns:
            tuple: (best_distance, best_order)
        """
        a = list(permutations(range(1, N)))
        last_best_distance = 1e10
        best_order = None

        for i in a:
            distance = 0
            pre_j = 0
            for j in i:
                distance = distance + w[j, pre_j]
                pre_j = j
            distance = distance + w[pre_j, 0]
            order = (0,) + i
            if distance < last_best_distance:
                best_order = order
                last_best_distance = distance
                print("order = " + str(order) + " Distance = " + str(distance))

        return last_best_distance, best_order

    def draw_tsp_solution(self, G, order, colors, pos):
        """
        Draw the TSP solution using networkx.

        Args:
            G: Graph
            order: Order of nodes in the solution
            colors: Colors for nodes
            pos: Positions of nodes
        """
        G2 = nx.DiGraph()
        G2.add_nodes_from(G)
        n = len(order)

        for i in range(n):
            j = (i + 1) % n
            G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])

        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(
            G2, node_color=colors, edge_color="b", node_size=600,
            alpha=0.8, ax=default_axes, pos=pos
        )
        edge_labels = nx.get_edge_attributes(G2, "weight")
        nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)

    def create_graph_from_matrix(self, matrix):
        """
        Create a networkx graph from a distance matrix.

        Args:
            matrix: Distance matrix

        Returns:
            nx.Graph: A complete graph with weights from the matrix
        """
        G = nx.Graph()
        n = len(matrix)

        # Add nodes
        for i in range(n):
            G.add_node(i)

        # Add edges with weights
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=matrix[i][j])

        return G

    def run_all_samples(self):
        """Run the brute force TSP algorithm on all samples and print results."""
        print("\n" + "=" * 50)
        print("TSP SAMPLES ANALYSIS")
        print("=" * 50)

        results = []

        for sample_name, adj_matrix in self.distance_matrices:
            print(f"\n{sample_name}")
            print("-" * len(sample_name))

            n = len(adj_matrix)
            best_distance, best_order = self.brute_force_tsp(adj_matrix, n)

            print(
                "Best order from brute force = "
                + str(best_order)
                + " with total distance = "
                + str(best_distance)
            )

            # Optional: Compare with known optimal solution
            if sample_name in self.optimal_tours:
                print(f"Known optimal tour: {self.optimal_tours[sample_name]}")
                print(f"Known optimal distance: {self.optimal_distances[sample_name]}")

            # Store results
            results.append((sample_name, best_order, best_distance))

        return results

    def visualize_solution(self, sample_index, use_circular_layout=True):
        """
        Visualize the TSP solution for a specific sample.

        Args:
            sample_index: Index of the sample (0-4)
            use_circular_layout: Whether to use circular layout for nodes
        """
        if sample_index < 0 or sample_index >= len(self.distance_matrices):
            raise ValueError(f"Sample index must be between 0 and {len(self.distance_matrices) - 1}")

        sample_name, adj_matrix = self.distance_matrices[sample_index]
        G = self.create_graph_from_matrix(adj_matrix)

        # Solve TSP
        n = len(adj_matrix)
        best_distance, best_order = self.brute_force_tsp(adj_matrix, n)

        # Prepare visualization
        if use_circular_layout:
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        node_colors = ['r', 'g', 'b', 'y', 'c']

        plt.figure(figsize=(10, 8))
        plt.title(f"{sample_name} - Optimal Tour\nDistance: {best_distance}")

        # Draw the solution
        self.draw_tsp_solution(G, best_order, node_colors, pos)

        # Convert node indices to letters for display
        node_mapping = {i: chr(65 + i) for i in range(n)}
        path_letters = ' → '.join([node_mapping[node] for node in best_order]) + f' → {node_mapping[best_order[0]]}'

        plt.figtext(0.5, 0.01, f"Path: {path_letters}", ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()

    def add_custom_distance_matrix(self, name, matrix):
        """
        Add a custom distance matrix.

        Args:
            name: Name of the sample
            matrix: Distance matrix (should be a numpy array)
        """
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Distance matrix must be square")

        self.distance_matrices.append((name, matrix))
        print(f"Added custom distance matrix '{name}' with shape {matrix.shape}")


if __name__ == "__main__":
    tsp_solver = Node5BruteforceForTravellingSalesman()
    tsp_solver.run_all_samples()

    # Visualize a specific sample (e.g., Sample 1)
    tsp_solver.visualize_solution(0)