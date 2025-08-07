import json
from collections import OrderedDict


def sort_json_by_key(data, target_key="experiment_data"):

    if isinstance(data, str):
        data = json.loads(data)

    result = data.copy()

    if target_key not in data:
        raise KeyError(f"Key '{target_key}' not found in data")

    target_data = data[target_key]

    def parse_key(key):
        parts = key.split('_')
        first_num = int(parts[0])
        second_num = int(parts[1]) if len(parts) > 1 else 0
        return (first_num, second_num)

    # Get keys and sort them
    keys = list(target_data.keys())
    sorted_keys = sorted(keys, key=parse_key)

    sorted_target_data = OrderedDict()
    for key in sorted_keys:
        sorted_target_data[key] = target_data[key]

    result[target_key] = sorted_target_data

    return result


def sort_json_from_file(input_file, output_file=None, target_key="experiment_data"):

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    _sorted_data = sort_json_by_key(data, target_key)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(_sorted_data, f, indent=2, ensure_ascii=False)
        print(f"Sorted data written to: {output_file}")

    return _sorted_data


if __name__ == "__main__":
    sorted_data = sort_json_from_file('saved_result/experiment_data.json', 'saved_result/experiment_data_sorted.json')