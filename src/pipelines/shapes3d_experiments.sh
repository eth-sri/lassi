#!/bin/bash

# Example usages:
# ./shapes3d_experiments.sh orientation_naive
# ./shapes3d_experiments.sh orientation_lassi

perturb=$1
shift

case "$perturb" in
orientation_naive)
  ./shapes3d_pipelines.sh e2e --dataset glow_3dshapes_latent_lmdb_correlated_orientation --cls_sigmas "5" --perturb orientation
  cd ../certification
  ./eval_with_ground_truth.sh \
    --fair_encoder "fair_encoder/gen_model=glow_3dshapes_latent_dset_glow_3dshapes_latent_lmdb_correlated_orientation/object_hue_cls_weight_1.0_adv_weight_0.0_recon_weight_0.0/linear_2048_1024_seed_42_epochs_5_batch_size_500_lr_0.001_use_bn_True_normalize_output_True" \
    --fair_classifier_name "fair_classifier_object_hue_epochs_1_batch_size_128_lr_0.001_cls_sigma_5.0" --cls_sigma 5.0 \
    --perturb orientation
  cd ../pipelines
  ;;

orientation_lassi)
  ./shapes3d_pipelines.sh e2e --dataset glow_3dshapes_latent_lmdb_correlated_orientation --cls_sigmas "1" --adv_loss_weight 0.1 --random_attack_num_samples 100 --perturb orientation
  cd ../certification
  ./eval_with_ground_truth.sh \
    --fair_encoder "fair_encoder/gen_model=glow_3dshapes_latent_dset_glow_3dshapes_latent_lmdb_correlated_orientation/object_hue_cls_weight_1.0_adv_weight_0.1_recon_weight_0.0_perturb_orientation_eps_1.0_attr_vec_avg_diff/linear_2048_1024_seed_42_epochs_5_batch_size_500_lr_0.001_use_bn_True_normalize_output_True_uniform_attack_num_samples_100" \
    --fair_classifier_name "fair_classifier_object_hue_epochs_1_batch_size_128_lr_0.001_cls_sigma_1.0" --cls_sigma 1.0 \
    --perturb orientation
  cd ../pipelines
  ;;

*)
  echo "Perturbing $perturb is not supported"
  ;;
esac
