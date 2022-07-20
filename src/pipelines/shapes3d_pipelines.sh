#!/bin/bash

# Example usages:

# Encoder pipeline:
# CUDA_VISIBLE_DEVICES=0 ./shapes3d_pipelines.sh encoder \
#   [--adv_loss_weight 0.1 --random_attack_num_samples 10] [--split valid]

# Classifier pipeline:
# ./shapes3d_pipelines.sh classifier "0.1,0.25,0.5..." \
#   --fair_encoder_name "fair_encoder/gen_model=glow_3dshapes/..." [--split valid]

# End-to-end pipeline:
# ./shapes3d_pipelines.sh e2e \
#   [--dataset glow_3dshapes_latent_lmdb_correlated_orientation] \
#   [--adv_loss_weight 0.1 --random_attack_num_samples 100] --cls_sigmas "0.1,0.25,0.5..." [--split valid]

pipeline_part=$1
shift

case "$pipeline_part" in
encoder)
  echo "Launch encoder pipeline"
  python encoder_pipeline.py \
    --epochs 5 --train_encoder_batch_size 500 --lr 0.001 --save_artefacts True --save_period 1  \
    --dataset glow_3dshapes_latent_lmdb_correlated_orientation \
    --input_representation latent --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_3dshapes --glow_n_flow 32 --glow_n_block 3 \
    --encoder_type linear --encoder_hidden_layers "2048,1024" \
    --classify_attributes object_hue --perturb orientation --perturb_epsilon 1 \
    --enc_sigma 0.65 --enc_alpha 0.01 --enc_n 10000 --enc_n0 10000 \
    --sampling_batch_size 10000 --certification_batch_size 320 --skip 320 --split test \
    "$@"
  ;;

classifier)
  sigmas_str=$1
  shift

  # Delimiter is comma (,)
  IFS=',' read -ra sigmas_arr <<< "$sigmas_str"
  for cls_sigma in "${sigmas_arr[@]}"
  do
    cls_sigma=$(echo $cls_sigma | xargs)
    echo "Launch classifier pipeline [cls_sigma = $cls_sigma]"
    python classifier_pipeline.py \
      --epochs 1 --train_classifier_batch_size 128 --lr 0.001 --save_artefacts True --save_period 1 \
      --dataset glow_3dshapes_latent_lmdb_correlated_orientation \
      --input_representation latent --image_size 64 --n_bits 5 \
      --gen_model_type Glow --gen_model_name glow_3dshapes --glow_n_flow 32 --glow_n_block 3 \
      --encoder_type linear --encoder_hidden_layers "2048,1024" \
      --classify_attributes object_hue --perturb orientation --perturb_epsilon 1 \
      --cls_sigma $cls_sigma \
      --cls_alpha 0.001 --cls_n 100000 --cls_n0 2000 \
      --enc_sigma 0.65 --enc_alpha 0.01 --enc_n 10000 --enc_n0 10000 \
      --sampling_batch_size 1000 --certification_batch_size 320 --skip 320 --split test \
      "$@"
  done
  ;;

e2e)
  python end_to_end_pipeline.py --run_only_one_seed True \
    --train_encoder_epochs 5 --train_encoder_batch_size 500 --lr 0.001 \
    --train_classifier_epochs 1 --train_classifier_batch_size 128 \
    --save_artefacts True --save_period 1  \
    --dataset glow_3dshapes_latent_lmdb_correlated_orientation \
    --input_representation latent --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_3dshapes --glow_n_flow 32 --glow_n_block 3 \
    --encoder_type linear --encoder_hidden_layers "2048,1024" \
    --classify_attributes object_hue --perturb orientation --perturb_epsilon 1 \
    --cls_alpha 0.001 --cls_n 100000 --cls_n0 2000 \
    --enc_sigma 0.65 --enc_alpha 0.01 --enc_n 10000 --enc_n0 10000 \
    --sampling_batch_size 1000 --certification_batch_size 320 --skip 320 --split test \
    "$@"
  ;;

*)
  echo "Pipeline part $pipeline_part not recognized and therefore not supported"
  ;;
esac
