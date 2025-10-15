#!/bin/bash
set -e
python etl.py --mode build_train --out data/train.parquet
python train_real.py --train data/train.parquet --model model.joblib
echo "Training finished, model.joblib created."
