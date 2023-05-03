import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

# 簡単なニューラルネットワークにおける勾配法を用いた実装
class simpleNet:
    def __init__(self):
        # ガウス分布による重み W の2×3行列のランダム数値での初期化
        self.W = np.random.randn(2, 3)
    
    # 推論処理
    def predict(self, x):
        return np.dot(x, self.W)
    
    # 損失関数の計算
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss

net = simpleNet()

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

# 損失関数
# numerical_gradient の定義式に合わせる形で関数の形を表現
def f(W):
    return net.loss(x, t)

# 損失関数のWによる変化を計算
dW = numerical_gradient(f, net.W)

print(dW)