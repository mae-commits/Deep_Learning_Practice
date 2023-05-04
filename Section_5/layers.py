import numpy as np 
from common.functions import *
from common.util import im2col, col2im

class Relu:
    # mask 関数の初期化
    def __init__(self):
        self.mask = None
        
    # 順伝播
    def forward(self, x):
        # mask 関数により、条件式を満たすか否かで True, False の boolean に変換 
        self.mask = (x <= 0)
        out = x.copy()
        # boolean で True であるインデックスのみ0に変換
        out[self.mask] = 0
        
        return out
    
    # 逆伝播
    def backward(self, dout):
        # 順伝播で計算済みの mask を用いて 0 に変換
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
class Sigmoid:
    # 初期化
    def __init__(self):
        self.out = None
    
    # 順伝播
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    # 逆伝播
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx

class Affine:
    # 各パラメータの初期化
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        
        # 重み。バイアスパラメータの微分
        self.dW = None
        self.db = None
        
    # 順伝播
    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        
        out = np.dot(self.x, self.W) + self.b
        
        return out
    
    # 逆伝播
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        # 入力データの形状に戻す（テンソル対応）
        dx = dx.reshape(*self.original_x_shape)
        
        return dx
    
class SoftmaxWithLoss:
    # 各パラメータの初期化
    def __init__(self):
        # 損失関数
        self.loss = None
        # softmax 関数の出力
        self.y = None
        # 教師データ
        self.t = None
        
    # 順伝播
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    # 逆伝播
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 教師データが one-hot-vector の場合
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
            
        return dx