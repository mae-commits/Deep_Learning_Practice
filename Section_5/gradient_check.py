import sys, os
# 親ディレクトリのダイルをインポートするための設定
sys.path.append(os.pardir)
import numpy as np 
from dataset.mnist import load_mnist
from two_layer_net import TwolayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwolayerNet(input_size=784, hidden_size=50, output_size=10)

# 各計算の計算誤差確認のためなので、少な目のデータ数で実験
x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 数値微分と誤差逆伝播法による勾配確認
for key in grad_numerical.keys():
    # 各パラメータの勾配に対して誤差を計算
    # 誤差逆伝播法の計算が合っているかどうかを確認
    diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key + ":" + str(diff))