import json
from modules.features_encoding import FeaturesEncoding
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier


SEED = 42
numeric_features = [
    'Age',
    'Height',
    'Weight',
    'FCVC',
    'NCP',
    'CH2O',
    'FAF',
    'TUE',
    'BMI'
    ]
FeatureScaler = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)


def bmi(df):
    data = df.copy()
    data['BMI'] = (data['Weight']/data['Height']**2)
    return data


with open('json/xgb.json', 'r') as json_file:
    params_xgb = json.load(json_file)
XGB = make_pipeline(
    FunctionTransformer(FeaturesEncoding),
    FeatureScaler,
    XGBClassifier(
        **{
            **params_xgb,
            **{
                'random_state': SEED,
                'tree_method': 'hist'
                }
            }
        )
)

with open('json/lgbm.json', 'r') as json_file:
    params_lgbm = json.load(json_file)
LGBM = make_pipeline(
    FunctionTransformer(FeaturesEncoding),
    FeatureScaler,
    LGBMClassifier(
        **{
            **params_lgbm,
            **{
                'random_state': SEED
                }
            }
        )
)

with open('json/cat.json', 'r') as json_file:
    params_cat = json.load(json_file)
CAT = make_pipeline(
    FunctionTransformer(bmi),
    FeatureScaler,
    CatBoostClassifier(
        **{
            **params_cat,
            **{
                'random_state': SEED,
                'thread_count': 4,
                'eval_metric': 'AUC',
                'loss_function': 'MultiClass',
                'verbose': False,
                'cat_features': [9, 10, 11, 12, 13, 14, 15, 16]
                }
            }
    )
)

with open('json/rf.json', 'r') as json_file:
    params_rf = json.load(json_file)
RF = make_pipeline(
    FunctionTransformer(FeaturesEncoding),
    FeatureScaler,
    RandomForestClassifier(
        **{
            **params_rf,
            **{
                'random_state': SEED
                }
            }
        )
)
