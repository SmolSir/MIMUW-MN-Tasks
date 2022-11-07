import numpy as np
import math

class dTridiag:
    def __str__(self):
        return f"dTridiag:\n \
            n:             {self.main_len}\n \
            d:             {self.dist}\n \
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
        self.dist = self.diagonal.size - self.hyperdiagonal.size
        self.__LU_decompose()

    def __LU_decompose(self):
        U_diagonal = np.copy(self.diagonal)
        U_hyperdiagonal = np.copy(self.hyperdiagonal)
        L_subdiagonal = np.copy(self.subdiagonal)
        for i in range(self.side_len):
            L_subdiagonal[i] /= U_diagonal[i]
            U_diagonal[i + self.dist] -= L_subdiagonal[i] * U_hyperdiagonal[i]
        self.U_diag = U_diagonal
        self.U_hyperdiag = np.concatenate((U_hyperdiagonal, np.zeros(self.dist)))
        self.L_subdiag = np.concatenate((np.zeros(self.dist), L_subdiagonal))

    def dot(self, v: np.ndarray):
        diag_comp = self.diagonal * v
        hyperdiag_comp = np.concatenate((self.hyperdiagonal * v[self.dist : ], np.zeros(self.dist)))
        subdiag_comp = np.concatenate((np.zeros(self.dist), self.subdiagonal * v[ : self.side_len]))
        return diag_comp + hyperdiag_comp + subdiag_comp

    def solve(self, y: np.ndarray):
        # solve L * temp = y
        L_subdiag = self.L_subdiag
        temp = np.zeros(self.main_len)
        y = y.astype(float)
        temp[0 : self.dist] = y[0 : self.dist]
        beg = self.dist
        while beg < self.main_len:
            end = min(beg + self.dist, self.main_len)
            y[beg : end] -= L_subdiag[beg : end] * temp[beg - self.dist : end - self.dist]
            temp[beg : end] = y[beg : end]
            beg = end

        # solve U * ans = temp
        # flip arrays to work top to bottom like the code above
        U_diag = np.flip(np.copy(self.U_diag))
        U_hyperdiag = np.flip(np.copy(self.U_hyperdiag))
        temp = np.flip(temp)
        ans = np.zeros(self.main_len)
        ans[0 : self.dist] = temp[0 : self.dist] / U_diag[0 : self.dist]
        beg = self.dist
        while beg < self.main_len:
            end = min(beg + self.dist, self.main_len)
            temp[beg : end] -= U_hyperdiag[beg : end] * ans[beg - self.dist : end - self.dist]
            ans[beg : end] = temp[beg : end] / U_diag[beg : end]
            beg = end
        return np.flip(ans)
