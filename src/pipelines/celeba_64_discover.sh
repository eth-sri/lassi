#!/bin/bash

# attr_vectors_discover (Li and Xu):

# - Naive:

./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0" --cls_sigmas "10" --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0,biased_attr_1" --cls_sigmas "10" --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0,biased_attr_1,biased_attr_2" --cls_sigmas "10" --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"

# - Data augmentation:

./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0" --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0,biased_attr_1" --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0,biased_attr_1,biased_attr_2" --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"

# - LASSI (cls + adv):

./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0" --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0,biased_attr_1" --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "biased_attr_0,biased_attr_1,biased_attr_2" --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_discover --perturb_epsilon 10 --enc_sigma 6.5 "$@"
