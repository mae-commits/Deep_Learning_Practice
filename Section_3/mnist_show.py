import sys, os
# 親ディレクトリのファイル読み込みを許可
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

# 画像データをPIL用データに変換
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 訓練データとテストデータの読み込み、及び配列を1次元に変換
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)

# データ形状を元のサイズに戻す
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
