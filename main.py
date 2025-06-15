import yaml
from src.Exterpolate import extrapolate, extrapolate_quantum
from src.RunAlgoWithDocplex import QuantumTSPRunner

config_file = "./config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
number_of_nodes_quantum = config['number_of_nodes_quantum']
output_dir = config.get('output_dir', '.saved_result')
result_directory = config['result_directory']

def main():
    runner = QuantumTSPRunner()
    runner.run_all_samples()

    plt = extrapolate_quantum(
                      '.saved_result/algo_with_docplex/quantum_tsp_results_final.pkl')

    # plt = extrapolate('.saved_result/algo_with_docplex/classical_tsp_results_final.csv', '.saved_result/algo_with_docplex/quantum_tsp_results_final.pkl')
    plt.savefig(f'{output_dir}/quantum_classical_comparison_improved.png', dpi=300, bbox_inches='tight')

    print("Process completed successfully!")

if __name__ == "__main__":
    main()