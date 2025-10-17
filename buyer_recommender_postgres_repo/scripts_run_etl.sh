#!/bin/bash
set -e
python etl.py --mode build_train --out data/train.parquet
python train_real.py --train data/train_ver_2.parquet --model model.joblib
echo "Training finished, model.joblib created."
