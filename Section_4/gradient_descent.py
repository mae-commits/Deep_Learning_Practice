from gradient_2d import numerical_gradient
import numpy as np
import matplotlib.pyplot as plt

def function_2(x):
    return x[0]**2 + x[1]**2

# 勾配降下法

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    next_x = init_x
    x = []
    # 更新値を配列に保存
    for i in range(step_num):
        x.append(next_x.copy())
        grad = numerical_gradient(f, next_x)
        next_x -= lr * grad
        
    return np.array(x)

if __name__ == '__main__':
    
    # 初期座標
    init_x = np.array([-3.0, 4.0])
    x = gradient_descent(function_2, init_x = init_x, lr = 0.1, step_num = 100)
    print(x[-1::])
    
    # 描画
    plt.plot( [-5, 5], [0,0], '--b')
    plt.plot( [0,0], [-5, 5], '--b')
    plt.figure()
    plt.plot(x[:,0], x[:,1], 'o')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.savefig('gradient_descent_2d.png')