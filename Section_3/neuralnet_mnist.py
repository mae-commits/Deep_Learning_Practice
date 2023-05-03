import sys, os
# 親ディレクトリのファイル読み込みを許可
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
from common.functions import sigmoid, softmax
import time

# データの読み込み
def get_data():
    # データの正規化(normalization): 画像のピクセル値を255で割り、正規化
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, t_test

# 推論処理に用いる重み・バイアスの読み込み
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
    return network

# 推論処理
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

start_time = time.time()
x, t = get_data()
network = init_network()

accuracy_cnt = 0

# 読み込んだ画像データの推論処理
for i in range(len(x)):
    y = predict(network, x[i])
    # 配列中の最大値のインデックスを出力
    p = np.argmax(y)
    # 正解ラベルと推定した数字が一致すれば正解カウントを１足す
    if p == t[i]:
        accuracy_cnt += 1

end_time = time.time()
print("Accuracy:" + str(float(accuracy_cnt)  / len(x)))
print("Calculation time:" + str(end_time-start_time))