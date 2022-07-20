#!/bin/bash

# - Naive:

./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "10" "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Young --enc_sigma 0.65 --cls_sigmas "10" "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "10" "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Heavy_Makeup --enc_sigma 0.65 --cls_sigmas "10" "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young" --enc_sigma 0.65 --cls_sigmas "10" "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --enc_sigma 0.65 --cls_sigmas "10" "$@"

./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "10" "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Young --enc_sigma 0.65 --cls_sigmas "10" "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "10" "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Heavy_Makeup --enc_sigma 0.65 --cls_sigmas "10" "$@"

# - Data augmentation:

./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Young --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Heavy_Makeup --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young" --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --train_encoder_batch_size 250 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 --train_encoder_batch_size 250 "$@"

./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Young --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Heavy_Makeup --enc_sigma 0.65 --cls_sigmas "10" --data_augmentation True --random_attack_num_samples 10 "$@"

# - LASSI (cls + adv):

./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Young --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb Heavy_Makeup --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young" --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Smiling --perturb "Pale_Skin,Young,Blond_Hair" --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"

./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Pale_Skin --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Young --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Blond_Hair --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.05 --random_attack_num_samples 10 "$@"
./celeba_pipelines_128.sh e2e --classify_attributes Wearing_Earrings --perturb Heavy_Makeup --enc_sigma 0.65 --cls_sigmas "2.5" --adv_loss_weight 0.25 --random_attack_num_samples 10 "$@"
