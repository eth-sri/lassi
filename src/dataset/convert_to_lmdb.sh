#!/bin/bash

# Example usage:
# ./convert_to_lmdb.sh glow_celeba_64
# ./convert_to_lmdb.sh glow_celeba_128
# ./convert_to_lmdb.sh glow_fairface
# ./convert_to_lmdb.sh glow_3dshapes

gen_model_name=$1
shift

case "$gen_model_name" in
glow_celeba_64)
  python convert_to_lmdb.py \
    --dataset celeba --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_celeba_64 \
    --glow_n_flow 32 --glow_n_block 4
  ;;

glow_celeba_128)
  python convert_to_lmdb.py \
    --dataset celeba --image_size 128 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_celeba_128 \
    --glow_n_flow 32 --glow_n_block 5
  ;;

glow_fairface)
  python convert_to_lmdb.py \
    --dataset fairface --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_fairface \
    --glow_n_flow 32 --glow_n_block 4
  ;;

glow_3dshapes)
  python convert_to_lmdb.py \
    --dataset 3dshapes --image_size 64 --n_bits 5 \
    --gen_model_type Glow --gen_model_name glow_3dshapes \
    --glow_n_flow 32 --glow_n_block 3 \
    "$@"
  ;;

*)
  echo "Generative model $gen_model_name not supported"
  ;;
esac
