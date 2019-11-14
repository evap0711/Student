import csv
from pandas import Series,DataFrame
import pandas as pd
from pylab import *
from pandas import *
import matplotlib.pyplot as plt
import numpy as np
from math import *
import glob

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

# 反復計算の上限回数
ITERATION_LIMIT = 1000

'''
Vincenty法(逆解法)
2地点の座標(緯度経度)から、距離と方位角を計算する
'''
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

#TwoWave model
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


# pythonフォルダ内にあるcsvファイルの一覧を取得
files = glob.glob("*.csv")
print(files)
# 全てのCSVファイルを読み込み、dictに入れる（keyはファイル名）
list = []
for file in files:
    list.append(pd.read_csv(file,encoding="utf-8_sig"))

df = concat(list, sort=False)

#学校の経緯度(固定値とする)
slon = 132.3443
slat = 34.369671
print("Start longitude :", slon)
print("Start latitude :", slat)
#直線からの範囲
erange = input("range(about 10 km per degree) :")
erange = float(erange)

#print(df.columns)

#device name(e0,d3,d4,d6)
df = df[ (df['devid']=='000015e0') | (df['devid']=='000015d3') | (df['devid']=='000015d6') | (df['devid']=='000015d4') ]
print("numbers",len(df))
xdata = df[' lon']
ydata = df[' lat']

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    
    if(event.button==3):#right click


        elon = float(event.xdata)
        elat = float(event.ydata)

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

        x = df[' lon']
        y = df[' lat']
        tnt = df[(df[' lon'] > lon_min) & (df[' lon'] < lon_max) & (df[' lat'] > lat_min) & (df[' lat'] < lat_max) & (abs(a*df[' lon']-1*df[' lat']+b)/(a**2+(-1)**2)**(1/2)<= erange)]

        xt = tnt[' lon']
        yt = tnt[' lat']
        rt = tnt[' rssi']

        #device name(e0,d3,d4,d6)
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



        #図1
        fig=plt.figure()
        ax1=fig.add_subplot(1,2,1)
        ax1.scatter(xt1,yt1,alpha=0.1,color="r",linewidths="1")
        ax1.scatter(xt3,yt3,alpha=0.1,color="g",linewidths="1")
        ax1.scatter(xt4,yt4,alpha=0.1,color="b",linewidths="1")
        ax1.scatter(xt6,yt6,alpha=0.1,color="b",linewidths="1")
        ax1.scatter(slon,slat,alpha=0.1,color="b",linewidths="1")
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

        ax2=fig.add_subplot(1,2,2)
        ax2.scatter(e0,rt1,alpha=0.1,color="r",linewidths="1")
        ax2.scatter(d3,rt3,alpha=0.1,color="g",linewidths="1")
        ax2.scatter(d4,rt4,alpha=0.1,color="y",linewidths="1")
        ax2.scatter(d6,rt6,alpha=0.1,color="b",linewidths="1")
        ax2.set_xlabel("distance")
        ax2.set_ylabel("rssi")
        ax2.set_title("e0,d3,d4,d6 : "+str(len(e0+d3+d4+d6)))

        #2波モデル適用
        xx=np.arange(0,max(d),1)
        yy=[]
        for i in xx:
            yy.append(TwoWave(i,56,1,920,4,10))
        ax2.plot(xx,yy)

        #e0
        fig1=plt.figure()
        ax3=fig1.add_subplot(1,2,1)
        ax3.scatter(xt1,yt1,alpha=0.1,color="r",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="b",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("e0:"+str(len(xt1)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig1.add_subplot(1,2,2)
        ax4.scatter(e0,rt1,alpha=0.1,color="r",linewidths="1")
        ax4.set_xlabel("distance")
        ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)

        #d3
        fig3=plt.figure()
        ax3=fig3.add_subplot(1,2,1)
        ax3.scatter(xt3,yt3,alpha=0.1,color="g",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="b",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("d3:"+str(len(xt3)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig3.add_subplot(1,2,2)
        ax4.scatter(d3,rt3,alpha=0.1,color="g",linewidths="1")
        ax4.set_xlabel("distance")
        ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)

        #d4
        fig4=plt.figure()
        ax3=fig4.add_subplot(1,2,1)
        ax3.scatter(xt4,yt4,alpha=0.1,color="y",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="b",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("d6:"+str(len(xt4)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig4.add_subplot(1,2,2)
        ax4.scatter(d6,rt6,alpha=0.1,color="y",linewidths="1")
        ax4.set_xlabel("distance")
        ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)

        #d6
        fig6=plt.figure()
        ax3=fig6.add_subplot(1,2,1)
        ax3.scatter(xt6,yt6,alpha=0.1,color="b",linewidths="1")
        ax3.scatter(slon,slat,alpha=0.1,color="b",linewidths="1")
        ax3.plot([slon,elon],[slat,elat])
        ax3.set_title("d6:"+str(len(xt6)))
        ax3.set_xlabel("longitude")
        ax3.set_ylabel("latitude")
        ax4=fig6.add_subplot(1,2,2)
        ax4.scatter(d6,rt6,alpha=0.1,color="b",linewidths="1")
        ax4.set_xlabel("distance")
        ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)

        #rssi
        #e0
        figx=plt.figure()
        ax1=figx.add_subplot(1,4,1)
        ax1.scatter(e0,rt1,alpha=0.1,color="r",linewidths="1")
        ax1.set_xlabel("distance")
        ax1.set_ylabel("rssi")
        ax1.plot(xx,yy)
        ax1.set_title("e0:"+str(len(e0)))
        #d3
        ax3=figx.add_subplot(1,4,2)
        ax3.scatter(d3,rt3,alpha=0.1,color="g",linewidths="1")
        ax3.set_xlabel("distance")
        ax3.set_ylabel("rssi")
        ax3.plot(xx,yy)
        ax3.set_title("d3:"+str(len(d3)))
        #d4
        ax4=figx.add_subplot(1,4,3)
        ax4.scatter(d4,rt4,alpha=0.1,color="y",linewidths="1")
        ax4.set_xlabel("distance")
        ax4.set_ylabel("rssi")
        ax4.plot(xx,yy)
        ax4.set_title("d4:"+str(len(d4)))
        #d6
        ax6=figx.add_subplot(1,4,4)
        ax6.scatter(d6,rt6,alpha=0.1,color="b",linewidths="1")
        ax6.set_xlabel("distance")
        ax6.set_ylabel("rssi")
        ax6.plot(xx,yy)
        ax6.set_title("d6:"+str(len(d6)))


        #graph show
        plt.show()
    
figb = plt.figure()
ax = figb.add_subplot(1, 1, 1)
ax.scatter(slon,slat,alpha=1,color="r",linewidths="2")
ax.scatter(xdata,ydata,alpha=0.1,color="b",linewidths="1")
figb.canvas.mpl_connect("button_press_event", onclick)
plt.show()