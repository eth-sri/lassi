import argparse
import copy

import args
from standard_classifier import eval_model, train_model
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset statistics and fairness evaluation for a standard model",
        parents=[
            args.common(),
            args.dataset(),
            args.gen_model(),
            args.glow(),
            args.encoder(),
            args.fair_encoder(),
            args.fair_classifier(),
            args.certification()
        ],
        conflict_handler='resolve'
    )
    parser.add_argument('--train_classifier_batch_size', type=int, required=True, help="Batch size for training")
    parser.add_argument('--eval_classifier_batch_size', type=int, required=True, help="Batch size for evaluation")
    parser.add_argument('--train_dataset', type=str, help="Dataset for training")
    parser.add_argument('--train_input_representation', type=str, help="Dataset representation for training")
    parser.add_argument('--compute_empirical_fairness', type=args.bool_flag, default=False,
                        help="Compute the empirical fairness of the classifier")
    parser.add_argument('--eval_dataset', type=str, help="Dataset for evaluation")
    parser.add_argument('--eval_input_representation', type=str, help="Dataset representation for evaluation")
    parser.add_argument('--run_only_one_seed', type=args.bool_flag, default=False,
                        help="Run only one seed instead of five.")
    return parser.parse_args()


if __name__ == '__main__':
    original_params = get_params()
    utils.init_logger()

    for seed in [42, 0, 10, 20, 30]:
        params = copy.deepcopy(original_params)
        params.seed = seed
        utils.set_random_seed(params.seed)

        # Train standard classifier:
        params.batch_size = params.train_classifier_batch_size
        if params.train_dataset is not None:
            assert params.train_input_representation is not None
            params.dataset = params.train_dataset
            params.input_representation = params.train_input_representation
        classifier_experiment_name = train_model.main(params)

        # Evaluate classifier:
        params.fair_classifier_name = classifier_experiment_name
        params.batch_size = params.eval_classifier_batch_size
        if params.eval_dataset is not None:
            assert params.eval_input_representation is not None
            params.dataset = params.eval_dataset
            params.input_representation = params.eval_input_representation
        eval_model.main(params)

        if params.run_only_one_seed:
            break
