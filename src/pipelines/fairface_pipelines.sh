#!/bin/bash

# Example usages:

# Encoder pipeline:
# CUDA_VISIBLE_DEVICES=0 ./fairface_pipelines.sh encoder \
#   --classify_attributes [Age|Age_bin|Age_3] --perturb Black \
#   [--adv_loss_weight 0.1 --random_attack_num_samples 5] \
#   [--contr_loss_weight 0.1 --contr_delta 10] \
#   [--encoder_type resnet --use_gen_model_reconstructions True --resnet_encoder_pretrained True]
# Computing centers arguments:
#   --enc_sigma [ENC_SIGMA] [--certification_batch_size 128 --skip 128 --splt valid]

# Classifier pipeline:
# ./fairface_pipelines.sh classifier "0.1,0.25,0.5..." \
#   --fair_encoder_name "fair_encoder/gen_model=glow_fairface/..." \
#   --classify_attributes [Age|Age_bin|Age_3] --perturb Black --enc_sigma [ENC_SIGMA]
#   [--encoder_type resnet --use_gen_model_reconstructions True] [--certification_batch_size 128 --skip 128 --split valid]
#   [--perform_full_analysis True]

# End-to-end pipeline:
# ./fairface_pipelines.sh e2e \
#   --classify_attributes [Age|Age_bin|Age_3] --perturb Black \
#   [--data_augmentation True --random_attack_num_samples 10]
#   [--adv_loss_weight 0.1 --random_attack_num_samples 10]
#   [--recon_loss_weight 0.1 --recon_decoder_type "linear" --recon_decoder_layers "1024,2048"]
#   [--encoder_type resnset --use_gen_model_reconstructions True]
#   --enc_sigma [ENC_SIGMA] --cls_sigmas "0.1,0.25,0.5..." \
#   [--certification_batch_size 128 --skip 128 --split valid]

pipeline_part=$1
shift

case "$pipeline_part" in
encoder)
  echo "Launch encoder pipeline"
  python encoder_pipeline.py \
    --epochs 5 --train_encoder_batch_size 500 --lr 0.001 --save_artefacts True --save_period 1  \
    --dataset glow_fairface_latent_lmdb --input_representation latent --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_fairface \
    --glow_n_flow 32 --glow_n_block 4 \
    --encoder_type linear --encoder_hidden_layers "2048,1024" \
    --perturb_epsilon 0.5 \
    --enc_alpha 0.01 --enc_n 10000 --enc_n0 10000 \
    --sampling_batch_size 10000 --certification_batch_size 32 --skip 32 --split test \
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
      --dataset glow_fairface_latent_lmdb --input_representation latent --image_size 64 --n_bits 5 \
      --gen_model_type Glow --gen_model_name glow_fairface \
      --glow_n_flow 32 --glow_n_block 4 \
      --encoder_type linear --encoder_hidden_layers "2048,1024" \
      --perturb_epsilon 0.5 \
      --cls_sigma $cls_sigma \
      --cls_alpha 0.001 --cls_n 100000 --cls_n0 2000 \
      --enc_alpha 0.01 --enc_n 10000 --enc_n0 10000 \
      --sampling_batch_size 1000 --certification_batch_size 32 --skip 32 --split test \
      "$@"
  done
  ;;

e2e)
  python end_to_end_pipeline.py \
    --train_encoder_epochs 5 --train_encoder_batch_size 500 --lr 0.001 \
    --train_classifier_epochs 1 --train_classifier_batch_size 128 \
    --save_artefacts True --save_period 1  \
    --dataset glow_fairface_latent_lmdb --input_representation latent --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_fairface \
    --glow_n_flow 32 --glow_n_block 4 \
    --encoder_type linear --encoder_hidden_layers "2048,1024" \
    --perturb_epsilon 0.5 \
    --cls_alpha 0.001 --cls_n 100000 --cls_n0 2000 \
    --enc_alpha 0.01 --enc_n 10000 --enc_n0 10000 \
    --sampling_batch_size 10000 --certification_batch_size 32 --skip 32 --split test \
    "$@"
  ;;

*)
  echo "Pipeline part $pipeline_part not recognized and therefore not supported"
  ;;
esac
