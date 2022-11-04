import numpy as np
import math

class dTridiag:
    def __str__(self):
        return f"n: {self.n}\n \
                 d: {self.d}\n \
                 diagonal:      {np.around(self.diagonal, decimals = 2)}\n \
                 hyperdiagonal: {np.around(self.hyperdiagonal, decimals = 2)}\n \
                 subdiagonal:   {np.around(self.subdiagonal, decimals = 2)}\n"

    def __init__(self, a:np.ndarray, b:np.ndarray, c:np.ndarray):
        self.n = a.size
        self.d = self.n - b.size
        self.diagonal = a
        self.hyperdiagonal = np.concatenate([b, np.zeros(self.d)])
        self.subdiagonal = np.concatenate([np.zeros(self.d), c])

    def dot(self, v: np.ndarray):
        pass

    def solve(self, y: np.ndarray):
        pass

test_dTridiag = dTridiag(np.array([1, 3, 6, 7]), np.array([2, 4]), np.array([5, math.pi]))
print(test_dTridiag)