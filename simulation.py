import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import requests
from PIL import Image
import math
zoom = 15

def get_tail_num(lat, lon, zoom):
    
    """
    緯度経度からタイル座標を取得する
    Parameters
    ----------
    lat : number 
        タイル座標を取得したい地点の緯度(deg) 
    lon : number 
        タイル座標を取得したい地点の経度(deg) 
    zoom : int 
        タイルのズーム率
    Returns
    -------
    xtile : int
        タイルのX座標
    ytile : int
        タイルのY座標
    """

    # https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def fetch_tile(z, x, y):
    url = "https://cyberjapandata.gsi.go.jp/xyz/dem5a/{z}/{x}/{y}.txt".format(z=z, x=x, y=y)
    try:
        df =  pd.read_csv(url, header=None,encoding="utf-8_sig").replace("e", 0.)
        df = df.astype(np.float)
        print("a")
        print(z,x,y)
    
    except:
        try:
            url = "https://cyberjapandata.gsi.go.jp/xyz/dem5b/{z}/{x}/{y}.txt".format(z=z, x=x, y=y)
            df =  pd.read_csv(url, header=None,encoding="utf-8_sig").replace("e", 0.)
            df = df.astype(np.float)
            print("b")
            print(z,x,y)
        except:
            df = np.zeros((256,256))
            print("nan")
            print(z,x,y)
            return df
    
    return df.values



def fetch_all_tiles(north_west, south_east):
    """ 北西端・南東端のタイル座標を指定して、長方形領域の標高タイルを取得 """
    assert north_west[0] == south_east[0], "タイル座標のzが一致していません"
    x_range = range(north_west[1], south_east[1]+1)
    y_range = range(north_west[2], south_east[2]+1)
    return  np.concatenate(
        [
            np.concatenate(
                [fetch_tile(north_west[0], x, y) for y in y_range],
                axis=0
            ) for x in x_range
        ],
        axis=1
    )
#34.381707 132.163403
#33.953656 132.688809
slat = input("Start latitube :")
slon = input("Start longitube :")
elat = input("End latitube :")
elon = input("End longitube :")
slat = float(slat)
slon = float(slon)
elat = float(elat)
elon = float(elon)
sx,sy = get_tail_num(slat,slon,zoom)
ex,ey = get_tail_num(elat,elon,zoom)
print(sx,sy)
print(ex,ey)

#tile = fetch_tile(15,28462,13057)
#tile = fetch_all_tiles((zoom, 28412, 13045), (zoom, 28448, 13085))
tile = fetch_all_tiles((zoom, sx, sy), (zoom, ex, ey))
print(tile.shape) # (512, 512)
# 可視化も簡単
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import *
fig = plt.figure()
ax = Axes3D(fig)
x,y = tile.shape
X,Y = np.meshgrid(np.linspace(slat,elat,y) , np.linspace(slon,elon,x))
ax.plot_surface(X,Y,tile)
plt.show()
