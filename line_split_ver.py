import numpy as np
import tifffile
from PIL import Image, ImageDraw
import math
import affine
import cv2
from utils import *
import os 
#print(os.getcwd())


####入出力pathの設定
path_anotation1 = ["./anotation/1/", "./anotation/2/", "./anotation/3/"]
path_anotation2 = ["./fish_img/1/", "./fish_img/2/", "./fish_img/3/"]
path_anotation_all = [path_anotation1, path_anotation2]

output_path_all = ["anotation_split_img/", "fish_split_img/"]
path_csv = "./input_csv/center_line_tai.csv"

####ハイパーパラメータ
n = 3 #分割数
targets = ["1", "2", "3"]

####csvファイルの読み込み
l_fish = read_csv(path_csv)

for path_anotation, output_path in zip(path_anotation_all, output_path_all):
    for target, path_ano in zip(targets, path_anotation):
        DIR = path_ano
        roop_num = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) +1 

        for num in range(1,roop_num):
            save_path = "./output/" + output_path + target + "/" + str(num) + "/" 
            my_makedirs(save_path)    ####フォルダの準備

            ####座標の読み取り
            if target == "1":
                x1 = int(l_fish[num][2])
                y1 = int(l_fish[num][3])
                x2 = int(l_fish[num][4])
                y2 = int(l_fish[num][5])
            if target == "2":
                x1 = int(l_fish[num+10][2])
                y1 = int(l_fish[num+10][3])
                x2 = int(l_fish[num+10][4])
                y2 = int(l_fish[num+10][5])
            if target == "3":
                x1 = int(l_fish[num+20][2])
                y1 = int(l_fish[num+20][3])
                x2 = int(l_fish[num+20][4])
                y2 = int(l_fish[num+20][5])

            ####画像の読み込み
            img = Image.open(path_ano + str(num) + ".tif")

            #centerline_csvから取得したxの値によって画像を反転するか決める(逆にした場合 座標も逆にする)
            if(x1 >= x2):
                img = np.array(img)
                img = np.fliplr(img)
                height, width, _ = img.shape
                img = Image.fromarray(img)
                x1, y1 = width-x1, y1
                x2, y2 = width-x2, y2


            begin_point = {'x':x1,'y':y1}
            end_point = {'x':x2,'y':y2}
            #print(begin_point, end_point)

            ####中心線の角度を取得
            radian = math.atan2(end_point['y'] - begin_point['y'], end_point['x'] - begin_point['x'])
            degree = radian * (180/math.pi)
            #print(degree)


            #draw = ImageDraw.Draw(img)
            #draw.line([(x1,y1),(x2,y2)],fill = "Red", width = 8)
            #img_show(img) #検証用
            img = np.array(img) #高さ，幅を取得
            height, width, _ = img.shape
            tifffile.imsave(save_path + "origin.tif",img)

            ####中心線の角度を0度にするように基準点を魚の頭の部分として，画像を回転
            affineMatrix = cv2.getRotationMatrix2D((begin_point['x'], begin_point['y']), degree, 1) #回転行列の取得
            img = cv2.warpAffine(img, affineMatrix, (width, height), flags=cv2.INTER_LANCZOS4, borderMode= cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            tifffile.imsave(save_path + "rotate.tif", img)
            #img_show(img) #検証用

            ####回転した後の中心線の始点と終点
            after_end_point = rotate_point(begin_point,end_point,degree)
            #print(begin_point,after_end_point)
            if(begin_point['y'] != after_end_point['y']):
                begin_point['y'] = after_end_point['y'] 

            ####中心線をn分割し，その真ん中の部分を持ってくる
            split_point = centerline_split(begin_point['x'], after_end_point['x'], n)
            height, width, _ = img.shape #回転した後の画像の高さ，幅を更新
            img_crop = img[:,split_point['x1']:split_point['x2']]
            # img_show(img_crop) #検証用   
            tifffile.imsave(save_path + "img_crop.tif",img_crop)
