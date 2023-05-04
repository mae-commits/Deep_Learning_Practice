# 乗算layer
class MulLayer:
    # 順伝播の入力値を保持するため、x, y の初期化
    def __init__(self):
        self.x = None
        self.y = None
    
    # 順伝播
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    # 逆伝播
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy

# 加算layer
class AddLayer:
    # 逆伝播で順伝播の入力値は必要ないので、初期化する必要なし
    def __init__(self):
        pass
    
    # 順伝播
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y
        
        return out
    
    # 逆伝播
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        
        return dx, dy