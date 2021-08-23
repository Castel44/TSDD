#!/bin/bash
export PYTHONPATH=$PYTHONPATH:${pwd}
export CUDA_VISIBLE_DEVICES=0

datasets=('RBF' 'MovRBF' 'fin_wine' 'fin_digits08' 'fin_digits17' 'phishing' 'fin_musk' 'fin_bank' ' fin_adult')

for i in "${datasets[@]}"; do
  echo $i
  python src/concept_drift/main_reduce_cov.py --dataset $i --disable_print --headless --init_seed 420 --n_runs 20
done


