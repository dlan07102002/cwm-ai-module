import argparse, os
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from model_utils import build_features_from_candidates

def train_from_parquet(parquet_path, model_out='model.joblib'):
    df = pd.read_parquet(parquet_path)
    print('Loaded', len(df), 'rows from', parquet_path)
    X = build_features_from_candidates(df)
    y = df['matched'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':64,'verbose':-1}
    bst = lgb.train(params, train_data, num_boost_round=200)
    # evaluate
    y_pred = bst.predict(X_test)
    print('AUC:', round(roc_auc_score(y_test, y_pred),4))
    # save model (joblib wrapper)
    joblib.dump(bst, model_out)
    print('Saved model to', model_out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='data/train.parquet')
    p.add_argument('--model', default='model.joblib')
    args = p.parse_args()
    train_from_parquet(args.train, model_out=args.model)
