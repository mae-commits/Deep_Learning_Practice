import sys, os
# 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np

# 2層のニューラルネットワークにおける計算の実行
class TwolayerNet:
    
    # 初期化の関数
    # input_size: 入力層のサイズ
    # hidden_size: 隠れ層のサイズ
    # output_size: 出力層のサイズ
    # weight_init_std: 重みの平準化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 推論処理の実行
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        # 1層目（隠れ層）への出力
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        # 2層目（出力層）への出力
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    # 損失関数（交差エントロピー関数）の計算
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    # 認識精度の計算
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) /float(x.shape[0])
        return accuracy
    
    # 勾配降下法の実行
    # x: 入力データ、t: 教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        # 別ファイル中の勾配降下法関数の実行による計算結果の出力
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads