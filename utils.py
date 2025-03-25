import csv
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import tifffile
import scipy
from radiomics import glszm
import SimpleITK as sitk

####csvモジュールを使ってCSVファイルから1行ずつ読み込む関数
def read_csv(path): 
    with open(path,encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
    return l

####pillow_show表示用関数
def img_show(img):
    img_pillow = Image.fromarray(img) #検証用
    img_pillow.show()

####フォルダ作成関数
def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

####座標を回転させる関数
def rotate_point(begin_point,end_point,degree):#Point:座標，degree:回転させたい角度
    d_rad = -1 * math.radians(degree) # 角度法を弧度法に変換
    x = int(end_point['x'] * math.cos(d_rad) - end_point['y'] * math.sin(d_rad) + begin_point['x'] - begin_point['x'] * math.cos(d_rad) + begin_point['y'] * math.sin(d_rad))
    y = int(end_point['x'] * math.sin(d_rad) + end_point['y'] * math.cos(d_rad) + begin_point['y'] - begin_point['x'] * math.sin(d_rad) - begin_point['y'] * math.cos(d_rad))
    rotate_Point = {'x':x,'y':y}
    return rotate_Point

####中心線をn分割させる関数
def centerline_split(x1, x2, n):
    line_width = x2 - x1
    if(n % 2 == 0): #偶数の場合，中心の位置からの長さ
        center = int(line_width / 2) + x1
        split_num = int(line_width/n)
        left_coord = center - int(split_num/2)
        right_coord = center + int(split_num/2)
    else: #奇数の場合，中心線を何分割したところの真ん中の長さ 
        split_num = int(line_width / n)
        a = n // 2 #split_numに掛け算する係数
        left_coord = x1 + a*split_num
        right_coord = x2 - a*split_num
    split_point = {'x1':left_coord,'x2':right_coord}
    return split_point

def img_extraction(img, begin_x, end_x, center_y, n_cut, save_path):
    ####fish->0, green back->255
    bgrLower = np.array([0, 255, 0])    # 抽出する色の下限(BGR)
    bgrUpper = np.array([0, 255, 0])    # 抽出する色の上限(BGR)
    img_bin = cv2.inRange(img, bgrLower, bgrUpper) # BGRからマスクを作成

    img_top = np.zeros_like(img_bin)
    img_bottom = np.zeros_like(img_bin)
    # img_show(img_bin) #検証用

    ####持ってきた画像の中心線の上と下を上から中心線までの距離1/2と下から中心線までの距離1/2の画像を持ってくる
    ##マスクの作成
    for position_x in range(begin_x+1, end_x+1):
        sum = 0
        position_top = center_y
        while (sum == 0):
            sum += img_bin[position_top, position_x]
            position_top -= 1

        y_diff_top = center_y - position_top
        img_top[position_top + 5:(center_y - int(y_diff_top * (1 - 1/n_cut))), position_x] = 255

        sum = 0
        position_bottom = center_y
        while (sum == 0):
            sum += img_bin[position_bottom, position_x]
            position_bottom += 1

        y_diff_bottom = position_bottom - center_y
        img_bottom[(center_y + int(y_diff_bottom * (1 - 1/n_cut))):position_bottom - 5, position_x] = 255

    ##マスクと魚の画像の重畳
    img_top_color = np.copy(img)
    img_top_color[img_top == 0] = [0, 255, 0]
    img_bottom_color = np.copy(img)
    img_bottom_color[img_bottom == 0] = [0, 255, 0]

    #img_show(img_top_color)
    #img_show(img_bottom_color)

    img_top_color_crop = np.zeros_like(img_top_color)
    img_bottom_color_crop = np.zeros_like(img_bottom_color)

    img_top_color_crop = img_top_color[center_y-400:center_y, begin_x:end_x+1]
    img_bottom_color_crop = img_bottom_color[center_y:center_y+500, begin_x:end_x+1]

    return img_top_color, img_bottom_color, img_top_color_crop, img_bottom_color_crop

def img_ROIs_extraction(img, sign, begin_x, end_x, center_y, ROI_size, ROI_n, save_path):
        ####fish->0, green back->255
    bgrLower = np.array([0, 255, 0])    # 抽出する色の下限(BGR)
    bgrUpper = np.array([0, 255, 0])    # 抽出する色の上限(BGR)
    img_bin = cv2.inRange(img, bgrLower, bgrUpper) # BGRからマスクを作成
    img_bin = cv2.bitwise_not(img_bin)

    # img_show(img_bin) #検証用

    ###ROIのx座標をランダムで決定
    x_random = np.random.randint((begin_x + ROI_size // 2), (end_x - ROI_size // 2 ), ROI_n)
    # print("begin_x, end_x, ROI:", begin_x, end_x, x_random)
    # print("shape:", img.shape)

    ###ROIを切り出す
    ROIs_list = []
    for position_x in x_random:

        ###上部と下部でそれぞれ初期値を設定
        if sign == "top":
            position_top = 0
            position_bottom = center_y
        else:
            position_top = center_y
            position_bottom = img.shape[0] - 1

        ###y軸方向の中点を探索
        sum = 0
        while (sum == 0):
            sum += img_bin[position_top, position_x]
            position_top += 1

        sum = 0
        while (sum == 0):
            sum += img_bin[position_bottom, position_x]
            position_bottom -= 1

        ROI_y_center = position_top + ((position_bottom - position_top) // 2)

        #検証用
        #print("x, ROI_y_top, ROI_y_bottom, ROI_y_center:", position_x ,position_top, position_bottom, ROI_y_center)

        ###ROIを切り出す
        ROI_cropped = img[(ROI_y_center - (ROI_size // 2)):(ROI_y_center + (ROI_size // 2)), (position_x - (ROI_size // 2)):(position_x + (ROI_size // 2))]

        ROIs_list.append(ROI_cropped)

        tifffile.imsave(save_path + "ROI_" + str(len(ROIs_list)) + ".tif", ROI_cropped)

    return ROIs_list

def maxGrayLevel(img_bin):
    max_gray_level = 0
    (height,width)=img_bin.shape
    print("画像の高さと幅は次のとおりです。height, width",height,width)

    for y in range(height):
        for x in range(width):
            if img_bin[y][x] > max_gray_level:
                max_gray_level = img_bin[y][x]
    print("max_gray_level:", max_gray_level)
    return max_gray_level + 1

def get_Glcm(input,d_x,d_y, gray_level):
    #inputはグレースケール
    srcdata=np.copy(input)
    ret = np.zeros((gray_level, gray_level))
    (height,width) = input.shape

    ##ペアのカウント(行列Gの作成)
    n_pairs = 0

    if d_y < 0:
        j_start = -d_y
        j_end = height
    else:
        j_start = 0
        j_end = height - d_y

    if d_x < 0:
        i_start = -d_x
        i_end = width
    else:
        i_start = 0
        i_end = width - d_x

    for j in range(j_start, j_end):
        for i in range(i_start, i_end):
            rows = srcdata[j][i]
            cols = srcdata[j+d_y][i+d_x]
            if rows != 255 and cols != 255:
                ret[rows][cols]+=1.0
                n_pairs += 1

    # print("n pairs:", n_pairs)

    # print(ret)

    return ret, n_pairs

def feature_computer(p, gray_level):
    # Enr:角度2次モーメント（エネルギー),テクスチャの均一性、またはピクセルペアの繰り返しを測定します。グレーレベル値の分布が一定または周期的である場合、高いエネルギーが発生します。	
    # Cnt:連続する画素間のグレーレベルの急激な変化を測定する。コントラストが高い画像は、空間周波数が高いのが特徴。
    # Crr:画像の線形依存性を測定します。高い相関値は、隣接するピクセルペアのグレーレベル間に直線的な関係があることを意味します。
    # Var:異質性の指標で、グレーレベル値が平均と異なる場合に分散が大きくなる。
    # Epy:エントロピ,ENT）は、画像に含まれる情報量のランダム性を測定し、画像の複雑さを示します。共起行列のすべての値が等しいか、ピクセル値が最大のランダム性を示す場合、エントロピーが最大になります。
    # Hom:均一性、テクスチャの明瞭さと規則性を反映しています。均一な画像であるほど値が高い。
    Enr=0.0
    Cnt=0.0
    Crr=0.0
    Var=0.0
    Epy=0.0
    Hom=0.0

    ux=0.0
    uy=0.0
    sigma_x=0.0
    sigma_y=0.0

    for i in range(gray_level):##平均求める
        Px = 0.0
        Py = 0.0
        for j in range(gray_level):
            Px += p[i][j]
            Py += p[j][i]
        ux += i*Px
        uy += i*Py
    
    for i in range(gray_level):##分散求める
        Px = 0.0
        Py = 0.0
        for j in range(gray_level):
            Px += p[i][j]
            Py += p[j][i]
        sigma_x += (i-ux)*(i-ux)*Px
        sigma_y += (i-uy)*(i-uy)*Py

    for i in range(gray_level):
        for j in range(gray_level):
            Enr+=p[i][j]*p[i][j]
            Cnt+=(i-j)*(i-j)*p[i][j]
            Crr+=i*j*p[i][j]
            Var+=(i-ux)*(i-ux)*p[i][j]
            #Hom+=p[i][j]/(1+np.abs(i-j))
            Hom+=p[i][j]/(1+(i-j)*(i-j))
            if p[i][j]>0.0:
                Epy+=p[i][j]*math.log(p[i][j])
    
    Crr=(Crr-(ux*uy))/(math.sqrt(sigma_x)*math.sqrt(sigma_y))

         
    return Enr,Cnt,Crr,Var,-Epy,Hom

##https://sourceexample.com/article/jp/cad76b26f0627fb94e787be4439f1e05/
def calc_glcm(img, gray_level, d, mode=None):
    if(mode=="WHOLE"):
        ####fish->0, green back->255
        bgrLower = np.array([0, 255, 0])    # 抽出する色の下限(BGR)
        bgrUpper = np.array([0, 255, 0])    # 抽出する色の上限(BGR)
        img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGRからマスクを作成
        # tifffile.imsave("./img_mask.tif", img_mask)

        ####グレースケールに変換
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Y = 0.299 * R + 0.587 * G + 0.114 * B
        # tifffile.imsave("./img_gray.tif", img_gray)

        ###0だけ別枠に分けるplan
        img_bin = img_gray//(256//gray_level) # [0:255]->[0:7]
        #tifffile.imsave("./img_bin_before.tif", img_bin)
        img_bin[img_mask == 255] = 255
        #tifffile.imsave("./img_bin.tif", img_bin)
    
    if(mode=="ROI"):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bin = img_gray//(256//gray_level) # [0:255]->[0:7]

    ##全方向対応にする45度ずつ
    glcm_0, n_pairs_0=get_Glcm(img_bin,d,0,gray_level)       
    glcm_180, n_pairs_180=get_Glcm(img_bin,-d,0,gray_level)

    glcm_45, n_pairs_45=get_Glcm(img_bin,d,d,gray_level)
    glcm_225, n_pairs_225=get_Glcm(img_bin,-d,-d,gray_level)

    glcm_90, n_pairs_90=get_Glcm(img_bin,0,d,gray_level)
    glcm_270, n_pairs_270=get_Glcm(img_bin,0,-d,gray_level)

    glcm_135, n_pairs_135=get_Glcm(img_bin,-d,d,gray_level)
    glcm_315, n_pairs_315=get_Glcm(img_bin,d,-d,gray_level)

    #glcm_all = glcm_0 + glcm_45 + glcm_90 + glcm_135 + glcm_180 + glcm_225 + glcm_270 + glcm_315
    glcm_0_180 = glcm_0 + glcm_180
    glcm_45_225 = glcm_45 + glcm_225
    glcm_90_270 = glcm_90 + glcm_270
    glcm_135_315 = glcm_135 + glcm_315

    ##ペア数で除算して確率算出(行列Pの作成)
    #n_pairs_all = n_pairs_0 + n_pairs_45 + n_pairs_90 + n_pairs_135 + n_pairs_180 + n_pairs_225 + n_pairs_270 + n_pairs_315
    n_pairs_0_180 = n_pairs_0 + n_pairs_180
    n_pairs_45_225 = n_pairs_45 + n_pairs_225
    n_pairs_90_270 = n_pairs_90 + n_pairs_270
    n_pairs_135_315 = n_pairs_135 + n_pairs_315

    #glcm = glcm_all / float(n_pairs_all)
    glcmP_0_180 = glcm_0_180/ float(n_pairs_0_180)
    glcmP_45_225 = glcm_45_225/ float(n_pairs_45_225)
    glcmP_90_270 = glcm_90_270/ float(n_pairs_90_270)
    glcmP_135_315 = glcm_135_315/ float(n_pairs_135_315)
   
    # print(glcm)
    Enr,Cnt,Crr,Var,Epy,Hom=feature_computer(glcmP_0_180, gray_level)
    glcm_feature_0_180 = ["0_180",Enr,Cnt,Crr,Var,Epy,Hom]

    Enr,Cnt,Crr,Var,Epy,Hom=feature_computer(glcmP_45_225, gray_level)
    glcm_feature_45_225 = ["45_225",Enr,Cnt,Crr,Var,Epy,Hom]

    Enr,Cnt,Crr,Var,Epy,Hom=feature_computer(glcmP_90_270, gray_level)
    glcm_feature_90_270 = ["90_270",Enr,Cnt,Crr,Var,Epy,Hom]

    Enr,Cnt,Crr,Var,Epy,Hom=feature_computer(glcmP_135_315, gray_level)
    glcm_feature_135_315 = ["135_315",Enr,Cnt,Crr,Var,Epy,Hom]
    
    return [glcm_feature_0_180, glcm_feature_45_225, glcm_feature_90_270, glcm_feature_135_315]

def calc_glcm_mean(img, gray_level, d, mode=None):
    if(mode=="WHOLE"):
        ####fish->0, green back->255
        bgrLower = np.array([0, 255, 0])    # 抽出する色の下限(BGR)
        bgrUpper = np.array([0, 255, 0])    # 抽出する色の上限(BGR)
        img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGRからマスクを作成
        # tifffile.imsave("./img_mask.tif", img_mask)

        ####グレースケールに変換
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Y = 0.299 * R + 0.587 * G + 0.114 * B
        # tifffile.imsave("./img_gray.tif", img_gray)

        ###0だけ別枠に分けるplan
        img_bin = img_gray//(256//gray_level) # [0:255]->[0:7]
        #tifffile.imsave("./img_bin_before.tif", img_bin)
        img_bin[img_mask == 255] = 255
        #tifffile.imsave("./img_bin.tif", img_bin)
    
    if(mode=="ROI"):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bin = img_gray//(256//gray_level) # [0:255]->[0:7]

    ##全方向対応にする45度ずつ
    glcm_0, n_pairs_0=get_Glcm(img_bin,d,0,gray_level)       
    glcm_180, n_pairs_180=get_Glcm(img_bin,-d,0,gray_level)

    glcm_45, n_pairs_45=get_Glcm(img_bin,d,d,gray_level)
    glcm_225, n_pairs_225=get_Glcm(img_bin,-d,-d,gray_level)

    glcm_90, n_pairs_90=get_Glcm(img_bin,0,d,gray_level)
    glcm_270, n_pairs_270=get_Glcm(img_bin,0,-d,gray_level)

    glcm_135, n_pairs_135=get_Glcm(img_bin,-d,d,gray_level)
    glcm_315, n_pairs_315=get_Glcm(img_bin,d,-d,gray_level)

    #glcm_all = glcm_0 + glcm_45 + glcm_90 + glcm_135 + glcm_180 + glcm_225 + glcm_270 + glcm_315
    glcm_0_180 = glcm_0 + glcm_180
    glcm_45_225 = glcm_45 + glcm_225
    glcm_90_270 = glcm_90 + glcm_270
    glcm_135_315 = glcm_135 + glcm_315

    ##ペア数で除算して確率算出(行列Pの作成)
    #n_pairs_all = n_pairs_0 + n_pairs_45 + n_pairs_90 + n_pairs_135 + n_pairs_180 + n_pairs_225 + n_pairs_270 + n_pairs_315
    n_pairs_0_180 = n_pairs_0 + n_pairs_180
    n_pairs_45_225 = n_pairs_45 + n_pairs_225
    n_pairs_90_270 = n_pairs_90 + n_pairs_270
    n_pairs_135_315 = n_pairs_135 + n_pairs_315

    #glcm = glcm_all / float(n_pairs_all)
    glcmP_0_180 = glcm_0_180/ float(n_pairs_0_180)
    glcmP_45_225 = glcm_45_225/ float(n_pairs_45_225)
    glcmP_90_270 = glcm_90_270/ float(n_pairs_90_270)
    glcmP_135_315 = glcm_135_315/ float(n_pairs_135_315)

    #glcm_all = (glcmP_0_180 + glcmP_45_225 + glcmP_90_270 + glcmP_135_315) / float(4)
   
    # print(glcm)
    Enr,Cnt,Crr,Var,Epy,Hom=feature_computer(glcmP_0_180, gray_level)
    glcm_feature_0_180 = [Enr,Cnt,Crr,Var,Epy,Hom]

    Enr,Cnt,Crr,Var,Epy,Hom=feature_computer(glcmP_45_225, gray_level)
    glcm_feature_45_225 = [Enr,Cnt,Crr,Var,Epy,Hom]

    Enr,Cnt,Crr,Var,Epy,Hom=feature_computer(glcmP_90_270, gray_level)
    glcm_feature_90_270 = [Enr,Cnt,Crr,Var,Epy,Hom]

    Enr,Cnt,Crr,Var,Epy,Hom=feature_computer(glcmP_135_315, gray_level)
    glcm_feature_135_315 = [Enr,Cnt,Crr,Var,Epy,Hom]

    #glcm_feature_0_360 = list(map(lambda x: x + glcm_feature_45_225 + glcm_feature_90_270 + glcm_feature_135_315, glcm_feature_0_180))
    glcm_feature_0_360 = (np.array(glcm_feature_0_180) + np.array(glcm_feature_45_225) + np.array(glcm_feature_90_270) + np.array(glcm_feature_135_315))/4.0
    #glcm_feature_0_360_list = list(map( lambda x:x/4.0,glcm_feature_0_360))
    glcm_feature_0_360_list = glcm_feature_0_360.tolist()
    #print(glcm_feature_0_360_list)
    
    return glcm_feature_0_360_list

def make_img_mask(img, gray_level):

    "gray_levelの調整した画像の作成"
    ##緑色の部分を白にそれ以外を黒に
    bgrLower = np.array([0, 255, 0])    # 抽出する色の下限(BGR)
    bgrUpper = np.array([0, 255, 0])    # 抽出する色の上限(BGR)
    img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGRからマスクを作成
    #tifffile.imsave("./img_mask.tif", img_mask)

    ####グレースケールに変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Y = 0.299 * R + 0.587 * G + 0.114 * B
    #tifffile.imsave("./img_gray.tif", img_gray)

    #0だけ別枠に分けるplan
    img_bin = img_gray//(256//gray_level) # [0:255]->[0:7]
    #tifffile.imsave("./img_bin_before.tif", img_bin)
    img_bin[img_mask == 255] = 0 #gray_levelを調整した画像が完成

    #mask画像(対象1, 背景0の画像を作成)
    mask2 = cv2.bitwise_not(img_mask)
    mask2[mask2 == 255] = 1

    return img_bin, mask2

def glszm_t(image, mask, BIN):
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    settings = {}
    #settings["binWidth"] = BIN
    settings["binCount"] = BIN
    
    # Show GLSZM features
    glszmFeatures = glszm.RadiomicsGLSZM(image, mask,**settings)
    glszmFeatures.enableAllFeatures()
    """
    print('Will calculate the following GLSZM features: ')
    for f in glszmFeatures.enabledFeatures.keys():
        print('  ', f)
        print(getattr(glszmFeatures, 'get%sFeatureValue' % f).__doc__)
    """
    #print('Calculating GLSZM features...')
    results = glszmFeatures.execute()

    name_list, value_list = [], []
    for key, value in results.items():
        #name_list.append(key)
        value_list.append(value)

    return value_list

def calculationg(img_lbp, img_mask, point):
    position = list(zip(*np.where((img_mask == 1))))
    lbp_list = []

    for coordinate in position:
        lbp = img_lbp[coordinate]
        lbp_list.append(lbp)

    hist = np.histogram(lbp_list, bins=point + 2, range=(0, point + 2))
    #print(hist[0])   
    skew = scipy.stats.skew(hist[0]) # 歪度
    #print(skew)
    kurtosis = scipy.stats.kurtosis(hist[0]) # 尖度
    #print(kurtosis)

    # plt.hist(lbp_list, bins=points + 2, range=(0, points + 2), color='red')
    # plt.show()

    ave = np.average(lbp_list)
    var = np.var(lbp_list)
    # print("ave", ave, "var", var)

    return ave, var, skew, kurtosis

