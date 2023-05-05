import numpy as np
import matplotlib.pyplot as plt

# 各種活性化関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x)), 'sigmoid'

def ReLU(x):
    return np.maximum(0, x), 'ReLU'

def tanh(x):
    return np.tanh(x), 'tanh'

# 1000 個のランダムなデータ
input_data = np.random.randn(1000, 100)
# 各隠れそうのノード（ニューロン）の数
node_num = 100
# 隠れ層が5層
hidden_layer_size = 5
# アクティベーションの結果の格納
activations = {}

# 第1層への入力
x = input_data

# 活性化関数の名前の保存
function = ""

# 標準偏差の選び方
sigma = [1, 0.01, np.sqrt(1.0 / node_num), np.sqrt(2.0 / node_num)]

for i in range(hidden_layer_size):
    # 第2層目以降への入力
    if i != 0:
        x = activations[i-1]
        
    # さまざまな標準偏差による初期値分布の変化    
    # w = np.random.randn(node_num, node_num) * sigma[0]
    # w = np.random.randn(node_num, node_num) * sigma[1]
    # w = np.random.randn(node_num, node_num) * sigma[2]
    w = np.random.randn(node_num, node_num) * sigma[3]
    
    # 推論処理
    a = np.dot(x, w)
    
    # 使用する活性化関数を変化
    # z, function = sigmoid(a)
    z, function = ReLU(a)
    # z, function = tanh(a)
    
    # 出力
    activations[i] = z
    
for i, a in activations.items():
    # 各層ごとのアクティベーション分布の変化を並べて描画
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0:
        plt.yticks([], [])
        plt.xlim(0.1, 1)
        plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.savefig(function + "_sigma=" + str(sigma[3]) + ".png")