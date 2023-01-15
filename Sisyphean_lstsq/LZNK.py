import math
import numpy as np
from scipy.linalg import solve_triangular

class Givens:
    def __str__(self):
        digits = 2
        return f"Givens: \
            \na:   {self.a} \
            \nb:   {self.b} \
            \npos: {self.pos}\n"


    # In this task, we always use Givens rotations to zero an element
    # directly below another one, thus we will only remember the pos
    # value (as the pos = row = column of the top left cos() position).
    def __init__(self, a, b, pos):
        self.a = a
        self.b = b
        self.pos = pos


    # Notice that transposing the Givens matrix only changes the signs
    # of sin() and the r is computed by squaring the a & b values. Thus
    # we can just change the sign of b and call the .T method (almost) like
    # we would on a normal matrix.
    def T(self):
        return Givens(self.a, -self.b, self.pos)


    # Multiply the passed matrix by the rotation from the left (Givens * matrix)
    def L_apply(self, matrix):
        r = math.sqrt(pow(self.a, 2) + pow(self.b, 2))
        c =  self.a / r
        s = -self.b / r

        print("L_apply: cos() | sin() = ", round(c, 4), "|", round(s, 4))

        upper_row = np.copy(matrix[self.pos, : ])
        lower_row = np.copy(matrix[self.pos + 1, : ])

        matrix[self.pos, : ] = upper_row * c - lower_row * s
        matrix[self.pos + 1, : ] = upper_row * s + lower_row * c
        return


    # Multiply the passed matrix by the rotation from the right (matrix * Givens)
    def R_apply(self, matrix):
        r = math.sqrt(pow(self.a, 2) + pow(self.b, 2))
        c =  self.a / r
        s = -self.b / r

        print("R_apply: cos() | sin() = ", round(c, 4), "|", round(s, 4))

        left_col  = np.copy(matrix[ : , self.pos])
        right_col = np.copy(matrix[ : , self.pos + 1])

        print("left_col:\n", left_col)
        print("right_col:\n", right_col)

        matrix[ : , self.pos] = left_col * c + right_col * s
        matrix[ : , self.pos + 1] = right_col * c - left_col * s

class LZNK:
    # Not required, but very useful to allow print-ing the class object.
    def __str__(self):
        digits = 2
        return f"LZNK: \
            \nA:\n {self.A.round(digits)}\n \
            \nQ:\n {self.Q.round(digits)}\n \
            \nR:\n {self.R.round(digits)}\n"


    def __init__(self, A: np.ndarray):
        self.A = A
        self.rows, self.cols = A.shape
        self.Q, self.R = np.linalg.qr(A, 'complete')
        return


    def addcol(self, col: np.ndarray):
        self.A = np.hstack((self.A, col.reshape(-1, 1)))
        self.R = np.hstack((self.R, col.reshape(-1, 1)))
        self.cols = self.cols + 1
        last_col = self.cols - 1
        rotations = []
        for row in range(self.rows - 2, self.cols - 2, -1):
            rotation = Givens(self.R[row][last_col], self.R[row + 1][last_col], row)
            rotation.L_apply(self.R)
            rotations.append(rotation)

        for rotation in reversed(rotations):
            print(rotation)
            rotation.T().R_apply(self.Q)
        return


    def delcol(self, col: np.ndarray):
        pass


    def lstsq(self, b: np.ndarray):
        return solve_triangular(self.R[: self.cols, :],
                                self.Q[: , : self.cols].T.dot(b))
