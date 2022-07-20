#!/bin/bash

# Example usage:
# ./eval_with_ground_truth.sh \
#   --fair_encoder_name [FAIR_ENCODER_NAME] \
#   --fair_classifier_name [FAIR_CLASSIFIER_NAME] \
#   --cls_sigma [CLS_SIGMA]

echo "Launch 3dshapes evaluation"
python eval_with_ground_truth.py \
  --dataset 3dshapes --input_representation original --image_size 64 --n_bits 5 \
  --gen_model_type Glow --gen_model_name glow_3dshapes --glow_n_flow 32 --glow_n_block 3 \
  --encoder_type linear --encoder_hidden_layers "2048,1024" \
  --classify_attributes object_hue --perturb orientation --perturb_epsilon 1 \
  --cls_alpha 0.001 --cls_n 100000 --cls_n0 2000 \
  --enc_sigma 0.65 --enc_alpha 0.01 --enc_n 10000 --enc_n0 10000 \
  --sampling_batch_size 1000 --certification_batch_size 320 --skip 320 --split test \
  "$@"
