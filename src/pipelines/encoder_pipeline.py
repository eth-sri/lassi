import argparse

import args
from certification import save_centers
from lassi_encoder import train_encoder
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encoder pipeline: 1) training and 2) computing and saving centers (by center smoothing)",
        parents=[
            args.common(),
            args.dataset(),
            args.gen_model(),
            args.glow(),
            args.encoder(),
            args.fair_encoder(),
            args.certification()
        ],
        conflict_handler='resolve'
    )
    return parser.parse_args()


if __name__ == '__main__':
    params = get_params()
    utils.common_experiment_setup(params.seed)

    assert params.train_encoder_batch_size is not None and params.certification_batch_size is not None

    # Train encoder:
    params.batch_size = params.train_encoder_batch_size
    encoder_experiment_name = train_encoder.main(params)

    # Save centers:
    params.fair_encoder_name = encoder_experiment_name
    params.batch_size = params.certification_batch_size
    save_centers.main(params)
