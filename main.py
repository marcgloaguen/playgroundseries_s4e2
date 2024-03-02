from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import LabelEncoder
from modules.model import XGB, LGBM, CAT, RF


def main(target: str):
    train = pd.read_csv('data/train.csv', index_col='id')
    X = train.copy()
    lb = LabelEncoder()
    y = lb.fit_transform(X.pop(target))
    SEED = 42
    SPLITS = 5
    SKF = StratifiedKFold(n_splits=SPLITS, random_state=SEED, shuffle=True)
    estimators = [
        ('xgb', XGB),
        ('lgbm', LGBM),
        ('cat', CAT),
        ('rf', RF)
    ]
    voting = VotingClassifier(estimators, voting='soft')
    scores = cross_val_score(
        voting,
        X,
        y,
        scoring='accuracy',
        cv=SKF,
        n_jobs=-1,
        verbose=20
        )
    print(f'Mean score : {np.mean(scores):.5f} ± {np.std(scores):.5f}')
    voting.fit(X, y)
    test = pd.read_csv('data/test.csv', index_col='id')
    submission = pd.read_csv("data/sample_submission.csv", index_col='id')
    submission.loc[:, target] = lb.inverse_transform(
        voting.predict(test)
        )
    name = dt.now().strftime("%Y%m%d_%H%M")
    submission.to_csv(f"submission/{name}.csv")
    print("Submission saved")


if __name__ == "__main__":
    main('NObeyesdad')
