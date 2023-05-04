import sys, os
sys.path.append(os.pardir)
import numpy as np 
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwolayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        
        # layer の作成
        self.layers = OrderedDict()
        # 1 層目の layer 作成
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # Relu 関数
        self.layers['Relu1'] = Relu()
        # 2 層目の layer 作成
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        # 出力層の layer 作成
        self.lastLayer = SoftmaxWithLoss()
    
    # 推論処理
    def predict(self, x):
        # 各層に対して順伝播で推論処理を実行
        # OrderedDict なので計算する順番は入力層から計算されている
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    # 損失関数の計算
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    # 認識精度の計算
    def accuracy(self, x, t):
        y = self.predict(x)
        # 出力に対して最大のインデックスを第1軸に対して特定
        y = np.argmax(y, axis=1)
        # 教師データの次元が1よりも大きな場合は軸を指定して最大値のインデックスを特定
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # x:入力データ, t:教師データ
    # 数値微分による勾配計算
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    # 誤差逆伝播法を活用した勾配計算
    def gradient(self, x, t):
        # 順伝播
        # 出力に対する逆伝播を行うには、一度順伝播による学習が必要
        self.loss(x, t)
        
        # 逆伝播
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        # 各層ごとの入力・重み付けを保存
        layers = list(self.layers.values())
        
        # 出力から辿っていくので、リスト反転
        layers.reverse()
        
        # 各 layer において逆伝播法による微分値の計算
        for layer in layers:
            dout =layer.backward(dout)
            
        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        return grads