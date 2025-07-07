from os import mkdir
from os.path import exists

import yaml
import logging
import sys
import datetime

from src.ProcessQaoa import process


config_file = "./config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

output_dir = config.get('output_dir', '.saved_result')
if not exists(output_dir):
    mkdir(output_dir)

num_cities_list = config['num_cities_list']
use_simulator = config['use_simulator']
penalty_weight = config['penalty_weight']
max_iter = config['max_iter']
shots = config['shots']


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'{output_dir}/app.log'),
    ]
)

logging.getLogger('qiskit').setLevel(logging.WARNING)

def main():
    logging.info(f'++++++++++++++++ Starting at {datetime.datetime.now()} +++++++++++++++++')

    for num_cities in num_cities_list:
        process(
            num_cities=num_cities,
            penalty_weight=penalty_weight,
            max_iter=max_iter,
            shots=shots,
            use_simulator = True)

if __name__ == "__main__":
    main()