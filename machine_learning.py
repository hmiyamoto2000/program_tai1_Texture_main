#xgboost, random_forest, lightgbmを使い，特徴量の重要度を可視化するサンプルコード

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pandas as pd
import yaml
import csv
from utils import *

###データのpathや機械学習のパラメータなど読み込み
with open("./programs/2-texture_analysis/2-Machine_Learning/params/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

###csvファイルからデータの読み込み
csv_path = config["input_data_root"] #csvファイルのpath
data = pd.read_csv(csv_path)

###データの整形
#比較したい項目のみ出す
features = [c for c in data.columns if c != "Species" and c != "num"]

#speciesを0,1の値に変更する(0:"1", 1:"1"and"2")
data['Species'] = data['Species'].replace(1,0)
data['Species'] = data['Species'].replace(2,1)
data['Species'] = data['Species'].replace(3,1)

#学習用データ
x_train = data[features]
y_train = data["Species"].values
print(x_train.shape, y_train.shape)

#検証用データ
x_test = x_train.copy()
y_test = y_train.copy()
print(x_test.shape, y_test.shape)

dataset = [x_train, y_train, x_test, y_test]

###モデルの作成と学習
xgboost_function(dataset, config)
random_forest_function(dataset, features, config)
lightgbm_function(dataset, features, config)