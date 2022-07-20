#!/bin/bash

# Example usage:

# ./compute_attr_vectors_glow.sh celeba64 [--rewrite False] \
#   [--computation_method perpendicular --epochs 3 --lr 0.001 --normalize_vectors True] OR
#   [--computation_method ramaswamy --epochs 3 --lr 0.001 --target Smiling]
# ./compute_attr_vectors_glow.sh celeba64_discover [--rewrite False]
# ./compute_attr_vectors_glow.sh celeba128 [--rewrite False]
# ./compute_attr_vectors_glow.sh fairface [--rewrite False]
# ./compute_attr_vectors_glow.sh 3dshapes [--rewrite False]

dset=$1
shift

case "$dset" in

celeba64)
  python compute_attr_vectors_glow.py \
    --batch_size 512 --dataset glow_celeba_64_latent_lmdb --image_size 64 \
    --attributes "Pale_Skin,Young,Blond_Hair,Heavy_Makeup" \
    --gen_model_type Glow --gen_model_name "glow_celeba_64" --glow_n_flow 32 --glow_n_block 4 \
    --rewrite True \
    "$@"
  ;;

celeba64_discover)
  cd ../standard_classifier
  ./celeba_64.sh train-cnn --classify_attributes Smiling
  cd ../dataset
  python compute_attr_vectors_glow.py \
    --batch_size 512 --dataset glow_celeba_64_latent_lmdb --image_size 64 \
    --gen_model_type Glow --gen_model_name "glow_celeba_64" --glow_n_flow 32 --glow_n_block 4 \
    --rewrite True \
    --computation_method discover --epochs 3 --lr 0.001 --target Smiling \
    --fair_classifier_name "std_cls/gen_model=glow_celeba_64/Smiling_resnet_seed_42_epochs_5_batch_size_500_lr_0.001_use_bn_True_pretrained_False" \
    --encoder_type resnet --encoder_normalize_output False \
    "$@"
  ;;

celeba128)
  python compute_attr_vectors_glow.py \
    --batch_size 512 --dataset glow_celeba_128_latent_lmdb --image_size 128 \
    --attributes "Pale_Skin,Young,Blond_Hair,Heavy_Makeup" \
    --gen_model_type Glow --gen_model_name "glow_celeba_128" --glow_n_flow 32 --glow_n_block 5 \
    --rewrite True \
    "$@"
  ;;

fairface)
  python compute_attr_vectors_glow.py \
    --batch_size 512 --dataset glow_fairface_latent_lmdb --image_size 64 --classify_attributes Race \
    --attributes "Black" \
    --gen_model_type Glow --gen_model_name "glow_fairface" --glow_n_flow 32 --glow_n_block 4 \
    --rewrite True \
    "$@"
  ;;

3dshapes)
  python compute_attr_vectors_glow.py \
    --batch_size 512 --dataset glow_3dshapes_latent_lmdb --image_size 64 \
    --attributes "orientation" \
    --gen_model_type Glow --gen_model_name "glow_3dshapes" --glow_n_flow 32 --glow_n_block 3 \
    --rewrite True \
    "$@"
  ;;

*)
  echo "Dataset mode $dset not recognized and thus not supported"
  ;;

esac
