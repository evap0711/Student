import csv
from pandas import Series,DataFrame
import pandas as pd
from pylab import *
from pandas import *
import matplotlib.pyplot as plt
import numpy as np
from math import *
import glob
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

# 楕円体
ELLIPSOID_GRS80 = 1 # GRS80
ELLIPSOID_WGS84 = 2 # WGS84

# 楕円体ごとの長軸半径と扁平率
GEODETIC_DATUM = {
    ELLIPSOID_GRS80: [
        6378137.0,         # [GRS80]長軸半径
        1 / 298.257222101, # [GRS80]扁平率
    ],
    ELLIPSOID_WGS84: [
        6378137.0,         # [WGS84]長軸半径
        1 / 298.257223563, # [WGS84]扁平率
    ],
}

# 海岸線のシェープファイルのpath
path_1 = 'japan_map/'
# 海岸線のある県番号(瀬戸内海のみ)
kens_no = [33, 34, 35, 37, 38]
#kens_no = [34, 35,  38]

# 反復計算の上限回数
ITERATION_LIMIT = 1000

#Vincenty法(逆解法)2地点の座標(緯度経度)から、距離と方位角を計算する
def vincenty_inverse(slon, slat, elon, elat, ellipsoid=None):

    # 差異が無ければ0.0を返す
    if slat == elat and slon == elon:
        return None

    # 計算時に必要な長軸半径(a)と扁平率(ƒ)を定数から取得し、短軸半径(b)を算出する
    # 楕円体が未指定の場合はGRS80の値を用いる
    a, ƒ = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))
    b = (1 - ƒ) * a

    φ1 = radians(slat)
    φ2 = radians(elat)
    λ1 = radians(slon)
    λ2 = radians(elon)

    # 更成緯度(補助球上の緯度)
    U1 = atan((1 - ƒ) * tan(φ1))
    U2 = atan((1 - ƒ) * tan(φ2))

    sinU1 = sin(U1)
    sinU2 = sin(U2)
    cosU1 = cos(U1)
    cosU2 = cos(U2)

    # 2点間の経度差
    L = λ2 - λ1

    # λをLで初期化
    λ = L

    # 以下の計算をλが収束するまで反復する
    # 地点によっては収束しないことがあり得るため、反復回数に上限を設ける
    for i in range(ITERATION_LIMIT):
        sinλ = sin(λ)
        cosλ = cos(λ)
        sinσ = sqrt((cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)
        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = atan2(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        cos2α = 1 - sinα ** 2
        cos2σm = cosσ - 2 * sinU1 * sinU2 / cos2α
        C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))
        λʹ = λ
        λ = L + (1 - C) * ƒ * sinα * (σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))

        # 偏差が.000000000001以下ならbreak
        if abs(λ - λʹ) <= 1e-20:
            break
    else:
        # 計算が収束しなかった場合はNoneを返す
        return None

    # λが所望の精度まで収束したら以下の計算を行う
    u2 = cos2α * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-1 + 2 * cos2σm ** 2) - B / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)))

    # 2点間の楕円体上の距離
    s = b * A * (σ - Δσ)

    return s

def Straight(x,y,x1,y1):#y=ax+b
    a = (y1 - y)/(x1 - x)
    b = (x1*y - x*y1)/(x1 - x)
    return a,b

def TwoWave(d,ht,hr,f,r,outd):
    lam = 3*10**2/f
    e0 = math.sqrt(d**2 + (ht - hr)**2)
    d2 = math.sqrt(d**2 + (ht + hr)**2)
    q = 2 * math.pi * (e0 - d2) / lam
    atan = math.atan(d/(ht + hr))
    angle = atan * 180 / math.pi
    cos = math.cos((90 - angle) * math.pi / 180)
    sin = math.sin((90 - angle) * math.pi / 180)
    R = (sin - math.sqrt(r - cos**2))/(sin + math.sqrt(r - cos**2))
    g = (lam/4/math.pi)**2 * ( (1/e0 + R * math.cos(q)/d2)**2 + (R * math.sin(q)/d2)**2)
    gain = 10 * math.log10(g) + outd
    return gain

def shapes(ken_no, num):
    
    if ken_no < 10 :
        ken_no_str = '0' + str(ken_no) # 県番号が10以下の場合、番号の前に"0"を入れる。
    else:
        ken_no_str = str(ken_no)
    dir = 'C23-06_' + ken_no_str + '_GML/' # 海岸線のシェープファイルが保存されているフォルダ名

    filename = 'C23-06_' + ken_no_str + '-g_Coastline.shp' # 海岸線のシェープファイル名

    # シェープファイルの表示
    fname = path_1 + dir + filename
    shapes = list(shpreader.Reader(fname).geometries())
    
    return shapes

# pythonフォルダ内にあるcsvファイルの一覧を取得
files = glob.glob("csv_file/*.csv")
print(files)
# 全てのCSVファイルを読み込み、dictに入れる（keyはファイル名）
dict = []
for file in files:
    dict.append(pd.read_csv(file,encoding="utf-8_sig"))

df = concat(dict, sort=False)

slon = 132.3443
slat = 34.369671

print("Start longitude :", slon)
print("Start latitude :", slat)

erange = input("range(about 10 km per degree) :")
einput = input("input type ( [0]GUI or [1]CUI) :")

erange = float(erange)
einput = float(einput)

#範囲指定（瀬戸内海周辺のみに限定）
df = df[(df[' lon'] > 132) & (df[' lon'] < 133) & (df[' lat'] > 33.5) & (df[' lat'] < 34.5) ]

#変更可
df = df[ (df['devid']=='000015e0') | (df['devid']=='000015d3') | (df['devid']=='000015d6') | (df['devid']=='000015d4') | (df['devid']=='00002ec3')]
print("numbers",len(df))
xdata = df[' lon']
ydata = df[' lat']


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
    
    if(event.button==3):#右クリック

        elon = float(event.xdata)
        elat = float(event.ydata)
        pltshow(elon,elat)

def pltshow(elon,elat):

    a,b = Straight(slon,slat,elon,elat)
    #範囲フィルタ
    if slat>elat:
        lat_max=slat
        lat_min=elat
    else:
        lat_max=elat
        lat_min=slat

    if slon>elon:
        lon_max=slon
        lon_min=elon
    else:
        lon_max=elon    
        lon_min=slon

    tnt = df[(df[' lon'] > lon_min) & (df[' lon'] < lon_max) & (df[' lat'] > lat_min) & (df[' lat'] < lat_max) & (abs(a*df[' lon']-1*df[' lat']+b)/(a**2+(-1)**2)**(1/2)<= erange)]
    
    if len(tnt) == 0:
        print('No Data in these spaces')
        return 0
    
    xt = tnt[' lon']
    yt = tnt[' lat']

    #変更可
    #e0,d3,d4,d6.2ec3
    e0=tnt[tnt['devid']=='000015e0']
    xt1 = e0[' lon']
    yt1 = e0[' lat']
    rt1 = e0[' rssi']

    d3=tnt[tnt['devid']=='000015d3']
    xt3 = d3[' lon']
    yt3 = d3[' lat']
    rt3 = d3[' rssi']

    d4=tnt[tnt['devid']=='000015d4']
    xt4 = d4[' lon']
    yt4 = d4[' lat']
    rt4 = d4[' rssi']

    d6=tnt[tnt['devid']=='000015d6']
    xt6 = d6[' lon']
    yt6 = d6[' lat']
    rt6 = d6[' rssi']

    e2=tnt[tnt['devid']=='00002ec3']
    xt2 = e2[' lon']
    yt2 = e2[' lat']
    rt2 = e2[' rssi']


    #図1
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1 = plt.axes(projection=ccrs.PlateCarree())
    for kens in kens_no :
        shape = shapes(kens, 1)
        ax1.add_geometries(shape, ccrs.PlateCarree(), edgecolor='black', facecolor='white', alpha=0.3)
    ax1.set_extent([132.2, 132.6, 33.97, 34.5])
    #ax1.scatter(xt,yt,alpha=0.1,color="r",linewidths="1")
    ax1.scatter(xt1,yt1,alpha=0.1,color="r",linewidths="1")
    ax1.scatter(xt3,yt3,alpha=0.1,color="g",linewidths="1")
    ax1.scatter(xt4,yt4,alpha=0.1,color="y",linewidths="1")
    ax1.scatter(xt6,yt6,alpha=0.1,color="b",linewidths="1")
    ax1.scatter(xt2,yt2,alpha=0.1,color="m",linewidths="1")
    ax1.scatter(slon,slat,alpha=0.1,color="k",linewidths="1")
    #y2 = (a * xt) + b
    ax1.plot([slon,elon],[slat,elat])
    ax1.set_title("("+str(elon)+" : "+str(elat)+")")
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")

    #図2
    d = []
    e0 = []
    d3 = []
    d4 = []
    d6 = []
    e2 = []
    for (xt,yt) in zip(xt, yt):
        distance = vincenty_inverse(slon, slat, xt, yt, 1)
        d.append(distance)

    #この下3つをfor(xt1,yt1)とすると元データのxt1,yt1が消えてしまうので注意
    for (xt,yt) in zip(xt1, yt1):
        distance1 = vincenty_inverse(slon, slat, xt, yt, 1)
        e0.append(distance1)

    for (xt,yt) in zip(xt3, yt3):
        distance3 = vincenty_inverse(slon, slat, xt, yt, 1)
        d3.append(distance3)
            
    for (xt,yt) in zip(xt4, yt4):
        distance4 = vincenty_inverse(slon, slat, xt, yt, 1)
        d4.append(distance4)

    for (xt,yt) in zip(xt6, yt6):
        distance6 = vincenty_inverse(slon, slat, xt, yt, 1)
        d6.append(distance6)

    for (xt,yt) in zip(xt2, yt2):
        distance2 = vincenty_inverse(slon, slat, xt, yt, 1)
        e2.append(distance2)

    figr = plt.figure()
    ax2 = figr.add_subplot(1, 1, 1)
    #ax2.scatter(d,rt,alpha=0.1,color="Blue",linewidths="1")
    def rssi(df, rt, c):
        ax2.scatter(df,rt,alpha=0.1,color=c,linewidths="1")
    names = ""
    if len(e0) != 0:
        rssi(e0, rt1, "r")
        names += "e0, "
    if len(d3) != 0:
        rssi(d3, rt3, "g")
        names += "d3, "
    if len(d4) != 0:
        rssi(d4, rt4, "y")
        names += "d4, "
    if len(d6) != 0:
        rssi(d6, rt6, "b")
        names += "d6, "
    if len(e2) != 0:
        rssi(e2, rt2, "m")
        names += "e2"

    ax2.set_xlabel("distance")
    ax2.set_ylabel("rssi")
    plt.subplots_adjust(wspace=0.4)
    ax2.set_title(names +  " : "+str(len(e0+d3+d4+d6+e2)))
    ax2.set_ylim([-160,-40])

    #2波モデル適用
    xx=np.arange(0,max(d),1)
    yy=[]
    for i in xx:
        yy.append(TwoWave(i,56,1,920,4,16))
    ax2.plot(xx,yy)
    
    #e0
    if len(e0) != 0:
        fig1=plt.figure()
        ax3 = fig1.add_subplot(1, 2, 1)
        ax3 = plt.axes(projection=ccrs.PlateCarree())
        for kens in kens_no :
            shape = shapes(kens, 1)
            ax3.add_geometries(shape, ccrs.PlateCarree(), edgecolor='black', facecolor='white', alpha=0.3)
        ax3.set_extent([132.2, 132.6, 33.97, 34.5])
        ax3.scatter(xt1,yt1,alpha=0.1,color="r",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="k",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("e0:"+str(len(xt1)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig1.add_subplot(1,2,2)
        ax4.scatter(e0,rt1,alpha=0.1,color="r",linewidths="1")
        ax4.set_xlabel("distance")
        #ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)
        ax4.set_ylim([-160,-40])

    #d3
    if len(d3) != 0:
        fig3=plt.figure()
        ax3 = fig3.add_subplot(1, 2, 1)
        ax3 = plt.axes(projection=ccrs.PlateCarree())
        for kens in kens_no :
            shape = shapes(kens, 1)
            ax3.add_geometries(shape, ccrs.PlateCarree(), edgecolor='black', facecolor='white', alpha=0.3)
        ax3.set_extent([132.2, 132.6, 33.97, 34.5])
        ax3.scatter(xt3,yt3,alpha=0.1,color="g",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="k",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("d3:"+str(len(xt3)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig3.add_subplot(1,2,2)
        ax4.scatter(d3,rt3,alpha=0.1,color="g",linewidths="1")
        ax4.set_xlabel("distance")
        #ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)
        ax4.set_ylim([-160,-40])

    #d4
    if len(d4) != 0:
        fig4=plt.figure()
        ax3 = fig4.add_subplot(1, 2, 1)
        ax3 = plt.axes(projection=ccrs.PlateCarree())
        for kens in kens_no :
            shape = shapes(kens, 1)
            ax3.add_geometries(shape, ccrs.PlateCarree(), edgecolor='black', facecolor='white', alpha=0.3)
        ax3.set_extent([132.2, 132.6, 33.97, 34.5])
        ax3.scatter(xt4,yt4,alpha=0.1,color="y",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="b",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("d4:"+str(len(xt4)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig4.add_subplot(1,2,2)
        ax4.scatter(d4,rt4,alpha=0.1,color="y",linewidths="1")
        ax4.set_xlabel("distance")
        #ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)
        ax4.set_ylim([-160,-40])

    #d6
    if len(d6) != 0:
        fig6=plt.figure()
        ax3 = fig6.add_subplot(1, 2, 1)
        ax3 = plt.axes(projection=ccrs.PlateCarree())
        for kens in kens_no :
            shape = shapes(kens, 1)
            ax3.add_geometries(shape, ccrs.PlateCarree(), edgecolor='black', facecolor='white', alpha=0.3)
        ax3.set_extent([132.2, 132.6, 33.97, 34.5])
        ax3.scatter(xt6,yt6,alpha=0.1,color="b",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="b",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("d6:"+str(len(xt6)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig6.add_subplot(1,2,2)
        ax4.scatter(d6,rt6,alpha=0.1,color="b",linewidths="1")
        ax4.set_xlabel("distance")
        #ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)
        ax4.set_ylim([-160,-40])

    #e2
    if len(e2) != 0:
        fig2=plt.figure()
        ax3 = fig2.add_subplot(1, 2, 1)
        ax3 = plt.axes(projection=ccrs.PlateCarree())
        for kens in kens_no :
            shape = shapes(kens, 1)
            ax3.add_geometries(shape, ccrs.PlateCarree(), edgecolor='black', facecolor='white', alpha=0.3)
        ax3.set_extent([132.2, 132.6, 33.97, 34.5])
        ax3.scatter(xt2,yt2,alpha=0.1,color="m",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="k",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("2ec3:"+str(len(xt2)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig2.add_subplot(1,2,2)
        ax4.scatter(e2,rt2,alpha=0.1,color="m",linewidths="1")
        ax4.set_xlabel("distance")
        #ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)
        ax4.set_ylim([-160,-40])

    #rssi比較
    figx=plt.figure()
    ax1=figx.add_subplot(1,5,1)
    ax1.scatter(e0,rt1,alpha=0.1,color="r",linewidths="1")
    ax1.set_xlabel("distance")
    #ax1.set_ylabel("rssi")
    ax1.plot(xx,yy)
    ax1.set_title("e0:"+str(len(e0)))
    ax1.set_ylim([-160,-40])

    ax2=figx.add_subplot(1,5,5)
    ax2.scatter(e2,rt2,alpha=0.1,color="m",linewidths="1")
    ax2.set_xlabel("distance")
    #ax2.set_ylabel("rssi")
    ax2.plot(xx,yy)
    ax2.set_title("2ec3:"+str(len(e2)))
    ax2.set_ylim([-160,-40])

    ax3=figx.add_subplot(1,5,2)
    ax3.scatter(d3,rt3,alpha=0.1,color="g",linewidths="1")
    ax3.set_xlabel("distance")
    #ax3.set_ylabel("rssi")
    ax3.plot(xx,yy)
    ax3.set_title("d3:"+str(len(d3)))
    ax3.set_ylim([-160,-40])

    ax4=figx.add_subplot(1,5,3)
    ax4.scatter(d4,rt4,alpha=0.1,color="y",linewidths="1")
    ax4.set_xlabel("distance")
    #ax4.set_ylabel("rssi")
    ax4.plot(xx,yy)
    ax4.set_title("d4:"+str(len(d4)))
    ax4.set_ylim([-160,-40])

    ax6=figx.add_subplot(1,5,4)
    ax6.scatter(d6,rt6,alpha=0.1,color="b",linewidths="1")
    ax6.set_xlabel("distance")
    #ax6.set_ylabel("rssi")
    ax6.plot(xx,yy)
    ax6.set_title("d6:"+str(len(d6)))
    ax6.set_ylim([-160,-40])

    #図表示
    plt.show()    

if(einput == 0):    
    figb = plt.figure(figsize=(7,10))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    ax = plt.axes(projection=ccrs.PlateCarree())
    for kens in kens_no :
        shape = shapes(kens, 1)
        ax.add_geometries(shape, ccrs.PlateCarree(), edgecolor='black', facecolor='white', alpha=0.3)
    ax.scatter(slon,slat,alpha=1,color="r",linewidths="2")
    ax.scatter(xdata, ydata, alpha=0.1, color="b", linewidths="1")
    figb.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
elif(einput == 1):
    elon = input("End longitude :")
    elat = input("End latitude :")
    elon = float(elon)
    elat = float(elat)
    pltshow(elon, elat)
else:
    print("Input Error")