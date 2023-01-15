from LZNK import LZNK
import numpy as np
import random


def small_lstsq_test():
    A = np.array([[1, 0, 2, 0],
                  [0, 3, 0, 4],
                  [5, 0, 6, 0],
                  [5, 0, 6, 0],
                  [0, 2, 0, 7],
                  [1, 3, 6, 7],
                  [1, 0, 5, 1]])

    c = np.array([1, 3, 6, 7, 4, 2, 1])
    b = np.array([1, 2, 3, 4, 4, 2, 1])
    m1 = LZNK(A)
    my_x = m1.lstsq(b)
    test_x = np.linalg.lstsq(A, b, rcond=None)[0]

    assert (np.allclose(my_x, test_x))

def small_addcol_test():
    A2 = np.array([[1, 0, 2, 0],
                   [0, 3, 0, 4],
                   [5, 0, 6, 0],
                   [5, 0, 6, 0],
                   [0, 2, 0, 7],
                   [1, 3, 6, 7],
                   [1, 0, 5, 1]])

    c2 = np.array([1, 3, 6, 7, 4, 2, 1])
    b2 = np.array([1, 2, 3, 4, 4, 2, 1])

    m2 = LZNK(A2)
    m2.addcol(c2)

    M2 = np.hstack((A2, c2.reshape(-1, 1)))
    my_x2 = m2.lstsq(b2)
    test_x2 = np.linalg.lstsq(M2, b2, rcond=None)[0]

    assert (np.allclose(my_x2, test_x2))


def small_delcol_test():
    A1 = np.array([[1, 0, 2, 0, 1],
                   [0, 3, 0, 4, 2],
                   [5, 0, 6, 0, 3],
                   [5, 0, 6, 0, 4],
                   [0, 2, 0, 7, 5],
                   [1, 3, 6, 7, 6],
                   [1, 0, 5, 1, 7]])

    b1 = np.array([1, 2, 3, 4, 4, 2, 1])

    m1 = LZNK(A1)

    column_to_delete = 1
    m1.delcol(column_to_delete)
    M1 = np.delete(A1, column_to_delete, 1)

    my_x1 = m1.lstsq(b1)
    test_x1 = np.linalg.lstsq(M1, b1, rcond=None)[0]

    assert (np.allclose(my_x1, test_x1))


def random_lstsq_test():
    for _ in range(1000):
        m = random.randint(3, 10)
        n = random.randint(m + 1, m + 10)
        A = np.random.rand(n, m)
        b = np.random.randint(100, size=(n))
        lznk = LZNK(A)
        assert np.allclose(np.linalg.lstsq(A, b, rcond=None)[0], lznk.lstsq(b))

def random_addcol_test():
    for _ in range(1000):
        m = random.randint(3, 10)
        n = random.randint(m + 1, m + 10)
        A = np.random.rand(n, m)
        c = np.random.randint(100, size=(n))
        b = np.random.randint(100, size=(n))
        lznk = LZNK(A)
        lznk.addcol(c)
        assert np.allclose(np.c_[A, c], np.dot(lznk.q, lznk.r))
        assert np.allclose(np.linalg.lstsq(np.c_[A, c], b, rcond=None)[0], lznk.lstsq(b))


def random_delcol_test():
    for _ in range(1000):
        m = random.randint(3, 10)
        n = random.randint(m + 1, m + 10)
        i = random.randint(0, m - 1)
        A = np.random.rand(n, m)
        b = np.random.randint(100, size=(n))
        lznk = LZNK(A)
        lznk.delcol(i)
        assert np.allclose(np.linalg.lstsq(np.delete(A, i, 1), b, rcond=None)[0], lznk.lstsq(b))


def add_then_delete_test():
    m = random.randint(3, 10)
    n = random.randint(m + 1, m + 10)
    i = m # on add it will be the last column
    A = np.random.rand(n, m)
    lznk = LZNK(A)
    c = np.random.randint(100, size=(n))
    b = np.random.randint(100, size=(n))
    lznk.addcol(c)
    assert np.allclose(np.c_[A, c], np.dot(lznk.q, lznk.r))
    assert np.allclose(np.linalg.lstsq(np.c_[A, c], b, rcond=None)[0], lznk.lstsq(b))
    lznk.delcol(i)
    assert np.allclose(np.linalg.lstsq(A, b, rcond=None)[0], lznk.lstsq(b))


def repeated_additions_test():
    m = random.randint(3, 10)
    n = random.randint(m + 10, m + 20)
    A = np.random.rand(n, m)
    for _ in range(5):
        c = np.random.randint(100, size=(n))
        b = np.random.randint(100, size=(n))
        lznk = LZNK(A)
        lznk.addcol(c)
        A = np.c_[A, c]
        assert np.allclose(A, np.dot(lznk.q, lznk.r))
        assert np.allclose(np.linalg.lstsq(A, b, rcond=None)[0], lznk.lstsq(b))


def repeated_deletion_test():
    m = random.randint(30, 40)
    n = random.randint(m + 10, m + 20)
    A = np.random.rand(n, m)
    for _ in range(5):
        i = random.randint(0, m)
        b = np.random.randint(100, size=(n))
        lznk = LZNK(A)
        lznk.delcol(i)
        assert np.allclose(np.linalg.lstsq(np.delete(A, i, 1), b, rcond=None)[0], lznk.lstsq(b))
        m -= 1


small_lstsq_test()
small_addcol_test()
small_delcol_test()
random_lstsq_test()
random_addcol_test()
random_delcol_test()
add_then_delete_test()
repeated_additions_test()
repeated_deletion_test()
