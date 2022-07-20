#!/bin/bash

# - Naive:

./fairface_pipelines.sh e2e --classify_attributes Age_bin --perturb Black --enc_sigma 0.325 --cls_sigmas "5" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_3 --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" "$@" # could not certify anything on the validation set, so at least keep the accuracy high
./fairface_pipelines.sh e2e --classify_attributes Age --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" "$@"

# - Data augmentation:

./fairface_pipelines.sh e2e --classify_attributes Age_bin --perturb Black --data_augmentation True --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "5" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_3 --perturb Black --data_augmentation True --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "2.5" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age --perturb Black --data_augmentation True --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "1" "$@"

# Transfer (cls = 0, adv=0.1 + recon=0.1)

./fairface_pipelines.sh e2e --classify_attributes Age_bin --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.1 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_3 --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.1 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./fairface_pipelines.sh e2e --classify_attributes Age --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" --train_encoder_epochs 20 --cls_loss_weight 0 --adv_loss_weight 0.1 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"

# Transfer (cls = 0.01 (Age_bin), adv=0.1 + recon=0.1)

./fairface_pipelines.sh e2e --train_encoder_classify_attributes Age_bin --train_classifier_classify_attributes Age_bin --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" --train_encoder_epochs 20 --cls_loss_weight 0.01 --adv_loss_weight 0.1 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./fairface_pipelines.sh e2e --train_encoder_classify_attributes Age_bin --train_classifier_classify_attributes Age_3 --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" --train_encoder_epochs 20 --cls_loss_weight 0.01 --adv_loss_weight 0.1 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"
./fairface_pipelines.sh e2e --train_encoder_classify_attributes Age_bin --train_classifier_classify_attributes Age --perturb Black --enc_sigma 0.325 --cls_sigmas "0.1" --train_encoder_epochs 20 --cls_loss_weight 0.01 --adv_loss_weight 0.1 --random_attack_num_samples 10 --recon_loss_weight 0.1 --recon_decoder_type linear --recon_decoder_layers 1024,2048 "$@"

# - LASSI (cls + adv):

./fairface_pipelines.sh e2e --classify_attributes Age_bin --perturb Black --adv_loss_weight 0.1 --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "0.25" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age_3 --perturb Black --adv_loss_weight 0.1 --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "0.25" "$@"
./fairface_pipelines.sh e2e --classify_attributes Age --perturb Black --adv_loss_weight 0.1 --random_attack_num_samples 10 --enc_sigma 0.325 --cls_sigmas "0.25" "$@"
