import numpy as np 
import matplotlib.pyplot as plt

# boolean 変換 -> int 型にすることで 0 or 1 の numpy 配列へ

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.savefig('step_function.png')