import argparse
import collections
import json

import numpy as np


SEEDS = [42, 0, 10, 20, 30]


def get_data_next_experiment(lines, seed):
    assert lines
    launch_prefix = 'Launch classifier pipeline [cls_sigma = '
    assert lines[0].startswith(launch_prefix)
    cls_sigma = lines[0][len(launch_prefix):-1]
    lines.popleft()
    if lines[0].startswith('{') and lines[0].endswith('}'):
        data = lines.popleft()
    else:
        data = None
    experiment_name = lines.popleft()
    assert f'cls_sigma_{cls_sigma}' in experiment_name
    assert f'seed_{seed}' in experiment_name
    maj_class = lines.popleft()
    if data is not None:
        assert lines.popleft() == data
    else:
        data = lines.popleft()
    return lines, experiment_name, maj_class, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyse the experiment output files.")
    parser.add_argument('--out_file', type=str, help="Experiment output file")
    parser.add_argument('--num_seeds', type=int, default=len(SEEDS), help="Number of seeds")
    params = parser.parse_args()

    with open(params.out_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if l]
        lines = collections.deque(lines)

    while lines:
        experiment_names = []
        maj_class_lines = []
        data = []
        for seed in SEEDS[:params.num_seeds]:
            lines, exp_name, maj_class, d = get_data_next_experiment(lines, seed)
            experiment_names.append(exp_name.replace(f'seed_{seed}', ''))
            maj_class_lines.append(maj_class)
            data.append(d)
        assert len(set(experiment_names)) == 1
        assert len(set(maj_class_lines)) == 1

        experiment_name = experiment_names[0]
        maj_class = maj_class_lines[0]

        print(experiment_name)
        print(maj_class)

        data = [json.loads(d.replace("'", '"')) for d in data]

        for d in data:
            print(d)

        acc_arr = [d['acc'] for d in data]
        if len(acc_arr) > 1:
            print(f'Acc : {np.mean(acc_arr):.1f} \pm {np.std(acc_arr, ddof=1):.1f}')
        else:
            print(f'Acc : {np.mean(acc_arr):.1f}')

        fair_arr = [d['fair'] if 'fair' in d else d['cert_fair'] for d in data]
        if len(fair_arr) > 1:
            print(f'Fair : {np.mean(fair_arr):.1f} \pm {np.std(fair_arr, ddof=1):.1f}')
        else:
            print(f'Fair : {np.mean(fair_arr):.1f}')
