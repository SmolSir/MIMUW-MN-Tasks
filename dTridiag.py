import numpy as np
import math

class dTridiag:
    def __str__(self):
        return f"dTridiag:\n \
            n:             {self.main_len}\n \
            d:             {self.d}\n \
            diagonal:      {np.around(self.diagonal, decimals = 4)}\n \
            hyperdiagonal: {np.around(self.hyperdiagonal, decimals = 4)}\n \
            subdiagonal:   {np.around(self.subdiagonal, decimals = 4)}\n \
            U_hyperdiag:   {np.around(self.U_hyperdiag, decimals = 4)}\n \
            U_diagonal:    {np.around(self.U_diag, decimals = 4)}\n \
            L_subdiagonal: {np.around(self.L_subdiag, decimals = 4)}\n"


    def __init__(self, a:np.ndarray, b:np.ndarray, c:np.ndarray):
        self.diagonal = a.astype(float)
        self.hyperdiagonal = b.astype(float)
        self.subdiagonal = c.astype(float)
        self.main_len = self.diagonal.size
        self.side_len = self.hyperdiagonal.size
        self.d = self.diagonal.size - self.hyperdiagonal.size
        self.__LU_decompose()


    def __LU_decompose(self):
        self.U_hyperdiag = np.concatenate((self.hyperdiagonal, np.zeros(self.d)))
        self.U_diag = np.copy(self.diagonal)
        self.L_subdiag = np.concatenate((np.zeros(self.d), self.subdiagonal))

        for i in range(self.side_len):
            self.L_subdiag[i + self.d] /= self.U_diag[i]
            self.U_diag[i + self.d] -= self.L_subdiag[i + self.d] * self.U_hyperdiag[i]


    def dot(self, v: np.ndarray):
        v = v.astype(float)
        result = self.diagonal * v
        result[ : self.side_len] += self.hyperdiagonal * v[self.d : ]
        result[self.d : ] += self.subdiagonal * v[ : self.side_len]
        return result


    def solve(self, y: np.ndarray):
        y = y.astype(float)
        temp = np.zeros(self.main_len)
        ans = np.zeros(self.main_len)

        # solve L * temp = y
        beg = self.d
        temp[ : beg] = y[ : beg]
        while beg < self.main_len:
            end = min(beg + self.d, self.main_len)
            y[beg : end] -= self.L_subdiag[beg : end] * temp[beg - self.d : end - self.d]
            temp[beg : end] = y[beg : end]
            beg = end

        # solve U * ans = temp
        end = self.main_len - self.d
        ans[end : self.main_len] = temp[end : self.main_len] / self.U_diag[end : self.main_len]
        while end > 0:
            beg = max(end - self.d, 0)
            temp[beg : end] -= self.U_hyperdiag[beg : end] * ans[beg + self.d : end + self.d]
            ans[beg : end] = temp[beg : end] / self.U_diag[beg : end]
            end = beg
        
        return ans
