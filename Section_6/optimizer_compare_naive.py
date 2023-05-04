import sys, os
# 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimizer import *

# 関数の設定
def f(x, y):
    return x**2 / 20.0 + y**2

# 関数の偏微分の設定
def df(x, y):
    return x / 10.0, 2.0 * y

# 初期位置
init_pos = (-7.0, 2.0)

# パラメータ
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]

# 勾配
grads = {}
grads['x'], grads['y'] = 0, 0

# 手法ごとに辞書型配列に格納
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=0.95)
optimizers['Momentum'] = Momentum(lr=0.1)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    # 手法選択
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    # 初期位置を設定
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        # 移動先の座標を追加
        x_history.append(params['x'])
        y_history.append(params['y'])
        # 勾配を計算
        grads['x'], grads['y'] = df(params['x'], params['y'])
        # 各手法によるパラメータの更新
        optimizer.update(params, grads)
    
    # 描画範囲の設定
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    # 2変数関数グラフを平面上に描くための設定
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # 一定以上の等高線を描かないように設定  
    mask = Z > 7
    Z[mask] = 0
    
    # 並べて描画
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.savefig('optimizer_compare_naive.png')