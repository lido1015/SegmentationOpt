import json
from typing import List, Tuple


def load_dataset(filepath: str = "data/dataset.json") -> List[Tuple[List[str], List[int]]]:
    """Load (contents, binary_cut_vector) from JSON."""

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = []
    for item in data:
        contents = item['contents']
        n = len(contents)
        binary = [0] * (n - 1)
        for idx in item['cuts']:
            binary[idx] = 1
        dataset.append((contents, binary))
    return dataset