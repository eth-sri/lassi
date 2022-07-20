#!/bin/bash

# Example usage (attr_vecs_avg_diff by default):

# ./celeba_64.sh train-cnn --classify_attributes Smiling
#   [--data_augmentation True --perturb Pale_Skin --cls_sigma [CLS_SIGMA]]

# ./celeba_64.sh eval-cnn --classify_attributes Smiling --perturb Pale_Skin \
#   --fair_classifier_name "std_cls/gen_model=glow_celeba_64/..."

# ./celeba_64.sh train-and-eval-cnn --classify_attributes Smiling --perturb Pale_Skin

# ./celeba_64.sh train-mlp --classify_attributes Smiling
#   [--data_augmentation True --perturb Pale_Skin --cls_sigma [CLS_SIGMA]]

mode=$1
shift

case "$mode" in
train-cnn)
  echo "Train standard ResNet/CNN model"
  python train_model.py \
    --epochs 5 --batch_size 500 --lr 0.001 --save_artefacts True --save_period 1 \
    --dataset celeba --input_representation original --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_celeba_64 --glow_n_flow 32 --glow_n_block 4 \
    --encoder_type resnet --encoder_normalize_output False --use_gen_model_reconstructions True \
    "$@"
  ;;

eval-cnn)
  echo "Evaluate standard ResNet/CNN model"
  python eval_model.py \
    --batch_size 64 --skip 64 --sampling_batch_size 0 \
    --dataset glow_celeba_64_latent_lmdb --input_representation latent --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_celeba_64 --glow_n_flow 32 --glow_n_block 4 \
    --encoder_type resnet --encoder_normalize_output False --use_gen_model_reconstructions True \
    --perturb_epsilon 1 \
    "$@"
  ;;

train-and-eval-cnn)
  echo "Train and evaluate standard ResNet/CNN model"
  python train_and_eval.py \
    --epochs 5 --train_classifier_batch_size 500 --lr 0.001 --save_artefacts True --save_period 1 \
    --eval_classifier_batch_size 64 --skip 64 --sampling_batch_size 0 \
    --train_dataset celeba --train_input_representation original \
    --eval_dataset glow_celeba_64_latent_lmdb --eval_input_representation latent \
    --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_celeba_64 --glow_n_flow 32 --glow_n_block 4 \
    --encoder_type resnet --encoder_normalize_output False --use_gen_model_reconstructions True \
    --perturb_epsilon 1 \
    "$@"
  ;;

train-mlp)
  echo "Train standard MLP model"
  python train_model.py \
    --epochs 5 --batch_size 500 --lr 0.001 --save_artefacts True --save_period 1 \
    --dataset glow_celeba_64_latent_lmdb --input_representation latent --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_celeba_64 --glow_n_flow 32 --glow_n_block 4 \
    --encoder_type linear --encoder_hidden_layers "2048,1024,512" \
    "$@"
  ;;

train-and-eval-mlp)
  echo "Train and evaluate standard MLP model"
  python train_and_eval.py \
    --epochs 5 --train_classifier_batch_size 500 --lr 0.001 --save_artefacts True --save_period 1 \
    --eval_classifier_batch_size 64 --skip 64 --sampling_batch_size 0 \
    --dataset glow_celeba_64_latent_lmdb --input_representation latent --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_celeba_64 --glow_n_flow 32 --glow_n_block 4 \
    --encoder_type linear --encoder_hidden_layers "2048,1024,512" \
    --perturb_epsilon 1 \
    "$@"
  ;;

*)
  echo "Mode $mode not recognized and therefore not supported."
  ;;
esac
