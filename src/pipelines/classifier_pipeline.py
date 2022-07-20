import argparse

import args
from certification import certify
from lassi_classifier import train_classifier
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classifier pipeline: 1) training and 2) get certification stats",
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
    return parser.parse_args()


if __name__ == '__main__':
    params = get_params()
    utils.common_experiment_setup(params.seed)

    assert params.train_classifier_batch_size is not None and params.certification_batch_size is not None

    # Train classifier:
    params.batch_size = params.train_classifier_batch_size
    classifier_experiment_name = train_classifier.main(params)

    # Run certification:
    params.fair_classifier_name = classifier_experiment_name.split('/')[-1]
    params.batch_size = params.certification_batch_size
    certify.main(params)
