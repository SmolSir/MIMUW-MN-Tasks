import numpy as np

class LZNK:
    # not required, but very useful to allow print-ing the class object
    def __str__(self):
        return f"LZNK:\n \
            A: {self.A}\n \
            Q: {self.Q}\n \
            R: {self.R}\n"


    def __init__(self, A: np.ndarray):
        self.A = A
        self.Q, self.R = np.linalg.qr(A, 'complete')

    def addcol(self, col: np.ndarray):
        pass

    def delcol(self, col: np.ndarray):
        pass

    def lstsq(self, b: np.ndarray):
        pass
