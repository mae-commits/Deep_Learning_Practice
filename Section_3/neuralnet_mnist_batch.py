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

x, t = get_data()
network = init_network()
start_time = time.time()

# バッチの数
batch_size = 100
accuracy_cnt = 0

# 読み込んだ画像データの推論処理をバッチサイズで区切って行う
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    # 配列中の最大値のインデックスを1次元目の要素（=各データ中の推定した数値）に応じて出力
    p = np.argmax(y_batch, axis=1)
    # 正解ラベルと推定した数字が一致している入力画像の数だけカウントを追加
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

end_time = time.time()    
print("Accuracy:" + str(float(accuracy_cnt)  / len(x)))
print("Calculation time:" + str(end_time-start_time))