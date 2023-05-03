import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def cross_entropy_error(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    # return -np.num(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

(x_train, t_train), (x_test, y_test) = load_mnist(normalize = True, one_hot_label = True)

train_size = x_train.shape[0]
batch_size = 10
# ランダムにbatch_size で設定した数だけデータのインデックスを指定
batch_mask = np.random.choice(train_size, batch_size)

# 指定したインデックスに読み込んだデータから選抜
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]