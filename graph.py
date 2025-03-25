from matplotlib import pyplot as plt
import japanize_matplotlib
import math
import os 
import yaml
import csv
import pandas as pd
import numpy as np
from utils import my_makedirs

with open("./programs/2-texture_analysis/3-graph/params/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

my_makedirs(config["output_data_root"])

#マークの大きさ
s = 100 

#csvファイルからデータの読み込み
csv_path1 = config["input_data_root1"] #csvファイルのpath
csv_path2 = config["input_data_root2"]
csv_path3 = config["input_data_root3"]
data1 = pd.read_csv(csv_path1)
data2 = pd.read_csv(csv_path2)
data3 = pd.read_csv(csv_path3)

#データの1列目と2列目を抜き出す
data1_1 = data1["Species"].values
data1_2 = data1["value"].values
data2_1 = data2["Species"].values
data2_2 = data2["value"].values
data3_1 = data3["Species"].values
data3_2 = data3["value"].values

#xgboostの上位何個のデータをrandomforestで探索
idx_list = []
idx_list2 = []
for elem1 in data1_1:
    idx = 0
    for elem2 in data2_1:
        if (elem1 == elem2):
            idx_list.append(idx)
        idx += 1

for elem1 in data1_1:
    idx = 0
    for elem2 in data3_1:
        if (elem1 == elem2):
            idx_list2.append(idx)
        idx += 1

new_data2_1 = []
new_data2_2 = []
for idx in idx_list:
    name = data2_1[idx]
    new_data2_1.append(name)
    value = data2_2[idx]
    new_data2_2.append(value)

new_data3_1 = []
new_data3_2 = []
for idx in idx_list2:
    name = data3_1[idx]
    new_data3_1.append(name)
    value = data3_2[idx]
    new_data3_2.append(value)
print(data1_2)
print(new_data3_1)
print(new_data3_2)

#csvファイルを作成
with open(config["output_data_root"]+"Important_feature.csv",'w',newline='') as f:
    writer = csv.writer(f)
    name_list = ["Species","XGboost_Importance","RandomForest_Importance","lightgbm_Importance"]
    writer.writerow(name_list)
    for name, data1, data2, data3 in zip(data1_1, data1_2, new_data2_2, new_data3_2):
        data = [name, data1, data2, data3]
        writer.writerow(data)

#exit()

#一度matplotlibでグラフ化して特徴量を比較(上位20個)
fig, ax = plt.subplots(1, 3, figsize=(50, 20))
ax[0].barh(data1_1[:20], data1_2[:20]) #xgboost
ax[1].barh(new_data2_1[:20], new_data2_2[:20]) #randomforest
ax[2].barh(new_data3_1[:20], new_data3_2[:20]) #lightgbm
ax[0].set_title("XGboost")
ax[1].set_title("RandomForest")
ax[2].set_title("lightgbm")
ax[0].set_xlabel("Feature importance")
ax[0].set_ylabel("Feature")
ax[1].set_xlabel("Feature importance")
ax[1].set_ylabel("Feature")
ax[2].set_xlabel("Feature importance")
ax[2].set_ylabel("Feature")
plt.savefig(config["output_data_root"] + "Compare_Important_features_top20.png", format="png")

#一度matplotlibでグラフ化して特徴量を比較(すべて)
fig, ax = plt.subplots(1, 3, figsize=(50, 50))
ax[0].barh(data1_1, data1_2) #xgboost
ax[1].barh(new_data2_1, new_data2_2) #randomforest
ax[2].barh(new_data3_1, new_data3_2) #lightgbm
ax[0].set_title("XGboost")
ax[1].set_title("RandomForest")
ax[2].set_title("lightgbm")
ax[0].set_xlabel("Feature importance")
ax[0].set_ylabel("Feature")
ax[0].set_ylabel("Feature")
ax[1].set_xlabel("Feature importance")
ax[1].set_ylabel("Feature")
ax[2].set_xlabel("Feature importance")
ax[2].set_ylabel("Feature")
plt.savefig(config["output_data_root"] + "Compare_Important_features_all.png", format="png")

#バブルチャートの作成

# figure オブジェクトを作成
# Create a figure object
fig = plt.figure()

# バブルチャートを作成
cm = plt.colormaps['rainbow'] #カラーマップ

scat = plt.scatter(data1_2[:20], new_data2_2[:20], s = s, c = new_data3_2[:20], cmap = cm, vmin=np.min(new_data3_2[:20]), vmax=np.max(new_data3_2[:20]), alpha = 0.5, label = 'control', marker="o")#, vmin = min_z, vmax = max_z)
fig.colorbar(scat)
plt.title("randomforest and xgboost and lightgbm")


# ラベルを追加
# Add labels
#plt.rcParams['font.family'] = 'Helvetica'
plt.xlabel("xgboost_importance")
plt.ylabel("randomforest_importance")

#　軸の追加
twin1 = plt.twinx()
twin1.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
twin1.tick_params(bottom=False, left=False, right=False, top=False)
twin1.set_ylabel("lightgbm_importance", labelpad=70)

plt.savefig(config["output_data_root"] + "baburu.png", format="png")