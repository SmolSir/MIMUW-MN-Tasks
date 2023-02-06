import numpy as np
import cmath

# Defining this as a const value is safer than typing the number 3 everywhere.
THREE = 3


# The main function of the task.
def parasolve(f, init, eps = 1e-3, N = 100):

    # Helper function - given x, check if |f(x)| <= eps.
    def __isclose(_x):
        return cmath.isclose(f(_x), complex(0), abs_tol = eps)

    # Helper function - given an array i = np.array([i_1, ..., i_n]) of indices
    # computes the divided difference f[x[i_1], x[i_2], ..., x[i_n]].
    def __div_diff(i: np.array):
        _n, _x, _fx = len(i), x[i], fx[i]
        for col in range(1, _n):
            _fx = (_fx[1 : ] - _fx[ : _n - col]) / (_x[col : ] - _x[ : _n - col])
        return _fx[0]

    # Helper function - compute the next term of the series in such a way that
    # division by zero is avoided (choose the non-zero denominator).
    def __next_term():
        denom_add, denom_sub = c + np.sqrt(delta), c - np.sqrt(delta)
        if denom_add != 0 and denom_sub != 0: # neither is zero
            x3_add = x[2] - (2 * __div_diff([2])) / denom_add
            x3_sub = x[2] - (2 * __div_diff([2])) / denom_sub

            x12_avg = (x[1] + x[2]) / complex(2)
            x3 = x3_add if np.abs(x12_avg - x3_add) < np.abs(x12_avg - x3_sub) \
                        else x3_sub

        if (denom_add == 0)  ^  (denom_sub == 0): # exactly one is zero
            if denom_sub == 0:
                x3 = x[2] - (2 * __div_diff([2])) / denom_add
            else:
                x3 = x[2] - (2 * __div_diff([2])) / denom_sub

        if (denom_add == 0) and (denom_sub == 0): # both are zero
            x3 = x[2] + complex(np.random.choice([-1, 1])) * fx[2]

        return x3

    # Assert that the provided init tuple is valid:
    assert len(init) == len(np.unique(init)) == THREE, \
        "The init tuple must contain three unique values."

    # These two arrays store the values of x_i & f(x_i), where x_i is equivalent
    # to x_{n + i} from the task's description. This allows us to write code
    # that is easier to both read and index arrays in.
    x  = np.array(init).astype(complex)
    fx = f(x)

    # Check if any of the init values sastisfies the condition |f(x)| <= eps:
    for _x in x:
        if __isclose(_x):
            return _x

    # We did not have to find the first 3 elements of the x_n series, so we are
    # left with at most N - 3 to iterate through.
    for _ in range(N - 3):

        # Compute the values of C, Delta, x3:
        c     = __div_diff([2, 1]) + __div_diff([2, 0]) - __div_diff([1, 0])
        delta = c ** 2 - 4 * __div_diff([2]) * __div_diff([2, 1, 0])
        x3 = __next_term()

        # Update the x & fx arrays by removing their oldest element & appending
        # the new value.
        x  = np.append(x[1 : ], [x3])
        fx = np.append(fx[1 : ], [f(x3)])

        if __isclose(x[2]):
            return x[2]

    print(f"No solution found after {N} iterations.")
    return x[2]
