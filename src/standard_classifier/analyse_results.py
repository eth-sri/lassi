import argparse
import collections
import json

import numpy as np


SEEDS = [42, 0, 10, 20, 30]


def print_stats(jsons):
    data = [json.loads(d) for d in jsons]
    for d in data:
        print(d)

    acc_arr = [d['acc'] for d in data]
    print(f'Acc : {np.mean(acc_arr):.1f} \pm {np.std(acc_arr, ddof=1):.1f}')

    if 'empirically_fair' in data[0]:
        fair_arr = [d['empirical_fair'] for d in data]
        print(f'Empirically Fair : {np.mean(fair_arr):.1f} \pm {np.std(fair_arr, ddof=1):.1f}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyse the experiment output files.")
    parser.add_argument('--out_file', type=str, help="Experiment output file")
    params = parser.parse_args()

    with open(params.out_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if l and l[0] == '{' and l[-1] == '}']
        lines = collections.deque(lines)

    while lines:
        valid_jsons = []
        test_jsons = []
        for _ in SEEDS:
            valid_jsons.append(lines.popleft())
            test_jsons.append(lines.popleft())

        print_stats(valid_jsons)
        print_stats(test_jsons)
