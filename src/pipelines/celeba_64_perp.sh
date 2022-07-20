#!/bin/bash

# attr_vectors_perp (Denton et al.):

# - Naive:

./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Pale_Skin --cls_sigmas "10" --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Young --cls_sigmas "10" --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Blond_Hair --cls_sigmas "10" --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Heavy_Makeup --cls_sigmas "10" --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young" --cls_sigmas "10" --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --cls_sigmas "10" --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"

# - Data augmentation:

./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Pale_Skin --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Young --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Blond_Hair --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Heavy_Makeup --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young" --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"

# - LASSI (cls + adv):

./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Pale_Skin --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Young --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Blond_Hair --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb Heavy_Makeup --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young" --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
./celeba_pipelines.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_perp --perturb_epsilon 10 --enc_sigma 6.5 "$@"
