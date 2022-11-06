import numpy as np
import math

class dTridiag:
    def __str__(self):
        return f"dTridiag:\n \
            n:             {self.n}\n \
            d:             {self.d}\n \
            diagonal:      {np.around(self.diagonal, decimals = 2)}\n \
            hyperdiagonal: {np.around(self.hyperdiagonal, decimals = 2)}\n \
            subdiagonal:   {np.around(self.subdiagonal, decimals = 2)}\n \
            U_hyperdiag:   {np.around(self.U_hyperdiag, decimals=2)}\n \
            U_diagonal:    {np.around(self.U_diag, decimals=2)}\n \
            L_subdiagonal: {np.around(self.L_subdiag, decimals=2)}\n"

    def __init__(self, a:np.ndarray, b:np.ndarray, c:np.ndarray):
        self.diagonal = a.astype(float)
        self.hyperdiagonal = b.astype(float)
        self.subdiagonal = c.astype(float)
        self.n = self.diagonal.size
        self.d = self.diagonal.size - self.hyperdiagonal.size
        self.__LU_decompose()

    def __LU_decompose(self):
        U_diagonal = self.diagonal
        U_hyperdiagonal = self.hyperdiagonal
        L_subdiagonal = self.subdiagonal
        for i in range(self.d):
            L_subdiagonal[i] /= U_diagonal[i]
            U_diagonal[i + self.d] -= L_subdiagonal[i] * U_hyperdiagonal[i]
        self.U_diag = U_diagonal
        self.U_hyperdiag = np.concatenate((U_hyperdiagonal, np.zeros(self.d)))
        self.L_subdiag = np.concatenate((np.zeros(self.d), L_subdiagonal))

    def dot(self, v: np.ndarray):
        diag_comp = self.diagonal * v
        hyperdiag_comp = np.concatenate((self.hyperdiagonal * v[self.d : ], np.zeros(self.d)))
        subdiag_comp = np.concatenate((np.zeros(self.d), self.subdiagonal * v[ : self.d]))
        return diag_comp + hyperdiag_comp + subdiag_comp

    def solve(self, y: np.ndarray):
        # solve L * temp = y
        L_subdiag = self.L_subdiag
        temp = np.zeros(self.n)
        temp[0 : self.d] = y[0 : self.d]
        beg = self.d
        while beg < self.n:
            end = min(beg + self.d, self.n)
            y[beg : end] -= L_subdiag[beg : end] * temp[beg - self.d : end - self.d]
            temp[beg : end] = y[beg : end]
            beg = end
        
        # solve U * ans = temp
        # flip arrays to reuse the code above
        U_diag = np.flip(self.U_diag)
        U_hyperdiag = np.flip(self.U_hyperdiag)
        ans = np.zeros(self.n)
        ans[0 : self.d] = temp[0 : self.d] / U_diag[0 : self.d]
        beg = self.d
        while beg < self.n:
            end = min(beg + self.d, self.n)
            

        ans[self.n - self.d : self.n] = temp[self.n - self.d : self.n] / U_diag[self.n - self.d : self.n]
        end = self.n - self.d



test_dTridiag = dTridiag(np.array([1, 3, 6, 7]), np.array([2, 4]), np.array([5, math.pi]))
print(test_dTridiag)
test_dTridiag.solve([1, 1, 1, 1])
test = np.array([1, 2, 3, 4, 5])