import numpy as np
from dTridiag import dTridiag

# TEST 1
# Matrix 4x4 d = 2, dot function
A1 = np.array([[1, 0, 2, 0],
             [0, 3, 0, 4],
             [5, 0, 6, 0],
             [0, np.pi, 0, 7]])

a1 = np.array([1, 3, 6, 7])
b1 = np.array([2, 4])
c1 = np.array([5, np.pi])

v1 = np.array([1, 2, 3, 4])

m1 = dTridiag(a1, b1, c1)

assert(np.allclose(m1.dot(v1), A1.dot(v1)))
print("TEST 1 passed! - example")


# TEST 2
# Matrix 4x4 d = 3, dot function
A2 = np.array([[1, 0, 0, 12],
             [0, 3, 0, 0],
             [0, 0, 6, 0],
             [11, 0, 0, 7]])

a2 = np.array([1, 3, 6, 7])
b2 = np.array([12])
c2 = np.array([11])

v2 = np.array([11, 2, 30, 4])

m2 = dTridiag(a2, b2, c2)

assert(np.allclose(m2.dot(v2), A2.dot(v2)))
print("TEST 2 passed! - small dot test 1")


# TEST 3
# Matrix 5x5 d = 2, dot function
A3 = np.array([[1, 0, 2, 0, 0],
             [0, 3, 0, 4, 0],
             [2, 0, 6, 0, 3],
             [0, 4, 0, 7, 0],
             [0, 0, 1, 0, 8]])

a3 = np.array([1, 3, 6, 7, 8])
b3 = np.array([2, 4, 3])
c3 = np.array([2, 4, 1])

v3 = np.array([5, 1, 15, 5, 25])

m3 = dTridiag(a3, b3, c3)

assert(np.allclose(m3.dot(v3), A3.dot(v3)))

print("TEST 3 passed! - small dot test 2")


# TEST 4
# Matrix 5x5 d = 2, multiple dot same matrix
repetitions = 100
A4 = np.array([[1, 0, 2, 0, 0],
             [0, 3, 0, 4, 0],
             [2, 0, 6, 0, 3],
             [0, 4, 0, 7, 0],
             [0, 0, 1, 0, 8]])

a4 = np.array([1, 3, 6, 7, 8])
b4 = np.array([2, 4, 3])
c4 = np.array([2, 4, 1])

m4 = dTridiag(a4, b4, c4)

for i in range (1, repetitions):
    v = np.repeat(i, a4.size)
    assert(np.allclose(m4.dot(v), A4.dot(v)))

print("TEST 4 passed! - multiple dot same matrix")


# TEST 5
# Matrix 5x5 d = 2, multiple solve (Ax = y) function same matrix
repetitions = 10
numRange = 3000
A5 = np.array([[1, 0, 2, 0, 0],
             [0, 3, 0, 4, 0],
             [2, 0, 6, 0, 3],
             [0, 4, 0, 7, 0],
             [0, 0, 1, 0, 8]])

a5 = np.array([1, 3, 6, 7, 8])
b5 = np.array([2, 4, 3])
c5 = np.array([2, 4, 1])

m5 = dTridiag(a5, b5, c5)

for i in range (1, repetitions):
    y = np.random.choice([-1, 1], a5.size) * np.random.uniform(0.001, numRange, a5.size)
    assert(np.allclose(np.linalg.solve(A5, y), m5.solve(y)))

print("TEST 5 passed! - multiple solve same matrix")


# TEST 6
# Matrix dot function random tests
numRange = 3000
matrixSize = 300
dRange = 100
for d in range(1, dRange):
    for n in range (d + 1, matrixSize - d):
        A = np.zeros((n, n))

        a = np.random.uniform(-numRange, numRange, n)
        b = np.random.uniform(-numRange, numRange, n - d)
        c = np.random.uniform(-numRange, numRange, n - d)
        v = np.random.randint(-numRange, numRange, n)

        m = dTridiag(a, b, c)

        for i in range(0, n):
            A[i][i] = a[i]
        for i in range(0, n - d):
            A[i + d][i] = c[i]
            A[i][i + d] = b[i]

        assert(np.allclose(A.dot(v), m.dot(v)))

print("TEST 6 passed! - random dot tests")


# TEST 7
# Matrix solve (Ax = y) function random tests
numRange = 3000
matrixSize = 300
dRange = 100
for d in range(1, dRange):
    for n in range (d + 1, matrixSize - d):
        A = np.zeros((n, n))

        a = np.random.choice([-1, 1], n) * np.random.uniform(0.001, numRange, n)
        b = np.random.choice([-1, 1], n - d) * np.random.uniform(0.001, numRange, n - d)
        c = np.random.choice([-1, 1], n - d) * np.random.uniform(0.001, numRange, n - d)
        y = np.random.choice([-1, 1], n) * np.random.uniform(0.001, numRange, n)

        m = dTridiag(a, b, c)

        for i in range(0, n):
            A[i][i] = a[i]
        for i in range(0, n - d):
            A[i + d][i] = c[i]
            A[i][i + d] = b[i]

        assert(np.allclose(np.linalg.solve(A, y), m.solve(y)))

print("TEST 7 passed! - random solve tests")