import numpy as np
import math

class dTridiag:
    def __str__(self):
        return f"dTridiag:\n \
            n:             {self.n}\n \
            d:             {self.d}\n \
            diagonal:      {np.around(self.diagonal, decimals = 2)}\n \
            hyperdiagonal: {np.around(self.hyperdiagonal, decimals = 2)}\n \
            subdiagonal:   {np.around(self.subdiagonal, decimals = 2)}\n"

    def __init__(self, a:np.ndarray, b:np.ndarray, c:np.ndarray):
        self.diagonal = a
        self.hyperdiagonal = b
        self.subdiagonal = c
        self.n = self.diagonal.size
        self.d = self.diagonal.size - self.hyperdiagonal.size

    def dot(self, v: np.ndarray):
        diag_comp = self.diagonal * v
        hyperdiag_comp = np.concatenate((self.hyperdiagonal * v[self.d : ], np.zeros(self.d)))
        subdiag_comp = np.concatenate((np.zeros(self.d), self.subdiagonal * v[ : self.d]))
        return diag_comp + hyperdiag_comp + subdiag_comp

    def solve(self, y: np.ndarray):
        pass

test_dTridiag = dTridiag(np.array([1, 3, 6, 7]), np.array([2, 4]), np.array([5, math.pi]))
print(test_dTridiag)
print(test_dTridiag.dot(np.array([1, 2, 3, 4])))