#!/bin/bash
export PYTHONPATH=$PYTHONPATH:${pwd}
export CUDA_VISIBLE_DEVICES=0



python src/concept_drift/main_RBF_drift.py --drift_features 'top' --drift_type 'flip' --abg 0 1 1 \
  --n_runs 10 --process 5 --disable_print --headless &&
python src/concept_drift/main_RBF_drift.py --drift_features 'bottom' --drift_type 'flip' --abg 0 1 1 \
  --n_runs 10 --process 5  --disable_print --headless &&
python src/concept_drift/main_RBF_drift.py --drift_features 'top' --drift_type 'flip' --abg 0 1 0 \
  --n_runs 10 --process 5 --disable_print --headless &&
python src/concept_drift/main_RBF_drift.py --drift_features 'bottom' --drift_type 'flip'  --abg 0 1 0 \
  --n_runs 10 --process 5 --disable_print --headless
