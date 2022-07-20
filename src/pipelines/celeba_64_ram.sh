#!/bin/bash

# attr_vectors_ram (Ramaswamy et al.):

# - Naive:

./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Pale_Skin --cls_sigmas "10" --attr_vectors_dir attr_vectors_ram --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Young --cls_sigmas "10" --attr_vectors_dir attr_vectors_ram --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Blond_Hair --cls_sigmas "10" --attr_vectors_dir attr_vectors_ram --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Heavy_Makeup --cls_sigmas "10" --attr_vectors_dir attr_vectors_ram --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "Pale_Skin,Young" --cls_sigmas "10" --attr_vectors_dir attr_vectors_ram --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --cls_sigmas "10" --attr_vectors_dir attr_vectors_ram --enc_sigma 0.65 "$@"

# - Data augmentation:

./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Pale_Skin --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Young --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Blond_Hair --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Heavy_Makeup --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "Pale_Skin,Young" --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"

# - LASSI (cls + adv):

./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Pale_Skin --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Young --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Blond_Hair --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb Heavy_Makeup --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "Pale_Skin,Young" --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
./celeba_pipelines.sh e2e --run_only_one_seed False --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 --attr_vectors_dir attr_vectors_ram --perturb_epsilon 1 --enc_sigma 0.65 "$@"
