#topとbottomに分けず，真ん中1/3を対象にして特徴量にする
##d 1~2 angle 0~135 各種特徴量をならべるようにする

import numpy as np
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import csv
import itertools
import tifffile
from radiomics import glszm
import SimpleITK as sitk
from skimage import feature
import scipy
import os
from utils import my_makedirs, calc_glcm_mean, make_img_mask, calculationg, glszm_t

####histgram_feature(やるか未定)

####GLCM_feature
#ハイパーパラメータとフォルダの準備
gray_level = 8 #正規化
mode = "WHOLE" #一枚の画像で算出
targets = ["1","2","3"]
distances = [1, 2] #画素間の距離
radiuses = [1, 2]
points = [8, 16]
BIN = 8
filename = "./output/2-texture_analysis/1-texture_feature/image_whole.csv"

def rearrange_elements(lst):
    rearranged_lst = []
    for item in lst:
        if len(item) >= 3:
            rearranged_item = item[-3:] + item[-4] + item[0:len(item)-4] 
            rearranged_lst.append(rearranged_item)
        else:
            rearranged_lst.append(item)
    return rearranged_lst

my_makedirs("./output/2-texture_analysis/1-texture_feature/")


###csvファイル一列目のname_listの作成
Spe_num = ["Species", "num"]
feature_name = ["GLCM_Enr", "GLCM_Con", "GLCM_Crr", "GLCM_Var", "GLCM_Epy", "GLCM_Hom"]
feature_name2 = ['GLSZM_GrayLevelNonUniformity', 'GLSZM_GrayLevelNonUniformityNormalized', 'GLSZM_GrayLevelVariance',\
                 'GLSZM_HighGrayLevelZoneEmphasis', 'GLSZM_LargeAreaEmphasis', 'GLSZM_LargeAreaHighGrayLevelEmphasis',\
                 'GLSZM_LargeAreaLowGrayLevelEmphasis', 'GLSZM_LowGrayLevelZoneEmphasis', 'GLSZM_SizeZoneNonUniformity',\
                 'GLSZM_SizeZoneNonUniformityNormalized', 'GLSZM_SmallAreaEmphasis', 'GLSZM_SmallAreaHighGrayLevelEmphasis',\
                 'GLSZM_SmallAreaLowGrayLevelEmphasis', 'GLSZM_ZoneEntropy', 'GLSZM_ZonePercentage', 'GLSZM_ZoneVariance']
feature_name3 = ['LBP_R1_P8_ave', 'LBP_R1_P8_var', 'LBP_R1_P8_skew', 'LBP_R1_P8_kurtosis',\
                 'LBP_R2_P16_ave', 'LBP_R2_P16_var', 'LBP_R2_P16_skew', 'LBP_R2_P16_kurtosis',]
distances_name = [f"d{name}" for name in distances]

all_name_list = []
for distance in zip(distances_name):
    combine_list = [distance, feature_name]
    combinations_list = list(itertools.product(*combine_list))
    name_list = ['_'.join(map(str, combination)) for combination in combinations_list]
    all_name_list.extend(name_list)
all_name_list = rearrange_elements(all_name_list)
all_name_list[0:0] = Spe_num #最初の2列にSpeciesとnumを追加
all_name_list = all_name_list + feature_name2 + feature_name3
#print(len(all_name_list), all_name_list)

#csvファイルを作成
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(all_name_list)


##control,testの解析をし，データをリストに追加していく
target_list = [] #最終格納配列

for target in targets:
    print("target", target)
    path = "./output/anotation_split_img/"+target+"/"
    directories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    roop_num = len(directories) + 1

    for num in tqdm(range(1,roop_num)):
        output_list = [target, num] #1列毎の格納配列

        for d in distances:
            top_f_list = [] #15*4*6要素
            input_path = "./output/anotation_split_img/"+target+"/"+str(num)+"/" 
            img = cv2.imread(input_path + "img_crop.tif")
            # symmetric = False
            # normed = True

            #glcmの特徴量
            g_f_list = calc_glcm_mean(img, gray_level, d, mode)
            output_list.extend(g_f_list)

        #glszmの特徴量
        img_level, mask = make_img_mask(img, gray_level)
        glszm_list = glszm_t(img_level, mask, BIN)
        output_list.extend(glszm_list)

        #lbpの特徴量
        img_gray, mask = make_img_mask(img, 256)
        for point, radius in zip(points, radiuses):
            # compute the Local Binary Pattern representation of the image
            lbp = feature.local_binary_pattern(img_gray, point, radius, method='uniform')
            if np.max(mask) != 0:
                ave, var, skew, kurtosis = calculationg(lbp, mask, point)
                lbp_features = [ave, var, skew, kurtosis]
                output_list.extend(lbp_features)
                #print("ave",ave,"var",var)
            else:
                ave, var, skew, kurtosis = 0, 0, 0, 0
                lbp_features = [ave, var, skew, kurtosis]
                output_list.extend(lbp_features)
        
        target_list.append(output_list)


#csvファイルへの書き込み(特徴量を1列ずつ追加)
with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(target_list)):
        output = target_list[i]
        writer.writerow(output)