import sys, os
# 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwolayerNet

# mnist手書きデータの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize =True, one_hot_label = True)

# 2層のニューラルネットワークにおける機械学習の関数の呼び出し
network = TwolayerNet(input_size = 784, hidden_size = 50, output_size = 10)

# 繰り返し回数の設定
iters_num = 10000
train_size = x_train.shape[0]
# バッチサイズの設定
batch_size = 100
# 学習率の設定
learning_rate = 0.1

# 各繰り返し計算における損失関数の値の記録
train_loss_list = []
# 訓練データでの各繰り返し計算における認識精度の値の記録
train_acc_list = []
# テストデータでの各繰り返し計算における認識精度の値の記録
test_acc_list = []

# epoch ごとに行う繰り返し計算の数の設定
iter_per_epoch = max(train_size / batch_size, 1)

# 繰り返し計算の実行
for i in range(iters_num):
    # 使用するデータのランダム選定
    batch_mask = np.random.choice(train_size, batch_size)
    # 使用する入力データと教師データ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 損失関数の計算
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # エポックごとに精度を計算・出力
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + "," + str(test_acc))
    
# グラフの描画 
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, label = 'train_acc')
plt.plot(x, test_acc_list, label = 'test_acc', linestyle = '--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc = 'lower right')
plt.savefig('train_neuralnet_graph.png')