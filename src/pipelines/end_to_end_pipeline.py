import argparse
import copy

import args
from certification import certify, save_centers
from lassi_classifier import train_classifier
from lassi_encoder import train_encoder
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: 1) training, 2) computing and saving centers (by center smoothing), "
                    "3) training downstream classifiers and 4) obtaining certification statistics",
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
    parser.add_argument('--train_encoder_epochs', type=int, required=True,
                        help="Number of training epochs for the encoder")
    parser.add_argument('--train_encoder_classify_attributes', type=args.argslist_str, default=[],
                        help="Classification task used to train the encoder")
    parser.add_argument('--train_classifier_epochs', type=int, required=True,
                        help="Number of training epochs for the classifier")
    parser.add_argument('--train_classifier_classify_attributes', type=args.argslist_str, default=[],
                        help="Transfer task on which the downstream classifier is trained")
    parser.add_argument('--cls_sigmas', type=args.argslist_float, required=True,
                        help="Comma-separated list of classifier sigmas")
    parser.add_argument('--run_only_one_seed', type=args.bool_flag, default=False,
                        help="Run only one seed instead of five.")
    return parser.parse_args()


if __name__ == '__main__':
    params = get_params()
    utils.init_logger()

    assert params.train_encoder_batch_size is not None
    assert params.train_classifier_batch_size is not None
    assert params.certification_batch_size is not None
    if params.train_encoder_classify_attributes:
        assert not params.classify_attributes
        assert params.train_classifier_classify_attributes
    else:
        assert params.classify_attributes
        assert not params.train_classifier_classify_attributes

    original_params = copy.deepcopy(params)

    for seed in [42, 0, 10, 20, 30]:
        params = copy.deepcopy(original_params)
        params.seed = seed
        utils.set_random_seed(params.seed)

        # Train encoder:
        params.epochs = params.train_encoder_epochs
        params.batch_size = params.train_encoder_batch_size
        if params.train_encoder_classify_attributes:
            params.classify_attributes = params.train_encoder_classify_attributes
        encoder_experiment_name = train_encoder.main(params)

        # Save centers:
        params.fair_encoder_name = encoder_experiment_name
        params.batch_size = params.certification_batch_size
        save_centers.main(params)

        params.epochs = params.train_classifier_epochs
        for cls_sigma in params.cls_sigmas:
            print(f'Launch classifier pipeline [cls_sigma = {cls_sigma}]')
            params.fair_classifier_name = None
            params.batch_size = params.train_classifier_batch_size
            params.cls_sigma = cls_sigma
            if params.train_classifier_classify_attributes:
                params.classify_attributes = params.train_classifier_classify_attributes

            # Train classifier:
            classifier_experiment_name = train_classifier.main(params)

            # Run certification:
            params.fair_classifier_name = classifier_experiment_name.split('/')[-1]
            params.batch_size = params.certification_batch_size
            certify.main(params)

        if params.run_only_one_seed:
            break
