import numpy as np

class LZNK:
    # Not required, but very useful to allow print-ing the class object.
    def __str__(self):
        digits = 2
        return f"LZNK: \
            \nA:\n {self.A.round(digits)}\n"


    def __init__(self, A: np.ndarray):
        self.A = A
        return

    def addcol(self, col: np.ndarray):
        self.A = np.hstack((self.A, col.reshape(-1, 1)))
        return

    def delcol(self, col: np.ndarray):
        self.A = np.delete(self.A, col, 1)
        return


    def lstsq(self, b: np.ndarray):
        return np.linalg.lstsq(self.A, b, rcond=None)[0]
