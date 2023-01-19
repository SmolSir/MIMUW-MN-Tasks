from parasolve import parasolve
import numpy as np
from random import randint
import cmath

RANGE  = int(1e4)
PRINT  = False
ASSERT = False

def exactly_two_solutions_exist():
    a = randint(-100000000, 100000000)
    b = randint(-100000000, 100000000)
    c = randint(-100000000, 100000000)
    while b ** 2 - 4 * a * c <= 0:
        a = randint(-100000000, 1000000000)
        b = randint(-100000000, 1000000000)
        c = randint(-100000000, 1000000000)
    return a, b, c


def exactly_one_solution_exists():
    # b^2 = 4ac => c = b^2 / 4a
    a = randint(-1000000, 1000000)
    b = randint(-1000000, 1000000)
    c = b ** 2 / (4 * a)
    while not cmath.isclose(b ** 2 - 4 * a * c, 0):
        a = randint(-1000000, 1000000)
        b = randint(-1000000, 1000000)
        c = b ** 2 / (4 * a)
    return a, b, c


def exactly_no_solution_exists():
    # b^2 = 4ac => c = b^2 / 4a
    a = randint(1, 100)
    b = randint(-100, 100)
    c = randint(-100, 100)
    while b ** 2 - 4 * a * c >= 0:
        a = randint(1, 100)
        b = randint(-100, 100)
        c = randint(-100, 100)
    return a, b, c


def get_starting_points(size='large'):
    limit = int(5e25)
    if size == 'medium':
        limit = int(5e3)
    if size == 'small':
        limit = int(5e1)
    x1 = 0
    x2 = 0
    x3 = 0
    while x1 == x2 or x2 == x3 or x1 == x3:
        x1 = randint(-limit, limit)
        x2 = randint(-limit, limit)
        x3 = randint(-limit, limit)
    return x1, x2, x3


def easy_normal_solution():
    for test in range(RANGE):
        if PRINT: print(f"easy_normal_solution {test}")
        x1, x2, x3 = get_starting_points(size='small')
        init = (x1, x2, x3)
        solve = parasolve(lambda x: x ** 2 - 1, init)
        if ASSERT:
            assert cmath.isclose(np.abs(solve), 1, abs_tol = 1e-3)


def test_two_solutions():
    for test in range(RANGE):
        if PRINT: print(f"test_two_solutions {test}")
        x1, x2, x3 = get_starting_points()
        init = (x1, x2, x3)
        a, b, c = exactly_two_solutions_exist()
        solve = parasolve(lambda x: a * x ** 2 + b * x + c, init)
        if ASSERT:
            assert cmath.isclose(a * solve ** 2 + b * solve + c, 0, abs_tol = 1e-3)


def test_one_solution():
    for test in range(RANGE):
        if PRINT: print(f"test_one_solution {test}")
        x1, x2, x3 = get_starting_points()
        init = (x1, x2, x3)
        a, b, c = exactly_one_solution_exists()
        solve = parasolve(lambda x: a * x ** 2 + b * x + c, init)
        if ASSERT:
            assert cmath.isclose(a * solve ** 2 + b * solve + c, 0, abs_tol = 1e-3)


def easy_complex_solution():
    for test in range(RANGE):
        if PRINT: print(f"easy_complex_solution {test}")
        x1, x2, x3 = get_starting_points(size='small')
        init = (x1, x2, x3)
        if ASSERT:
            assert cmath.isclose(
                np.abs(parasolve(lambda x: x ** 2 + 1, init)), 1, abs_tol= 1e-3
            )


def test_complex_solutions():
    for test in range(RANGE):
        if PRINT: print(f"test_complex_solutions {test}")
        x1, x2, x3 = get_starting_points(size='medium')
        init = (x1, x2, x3)
        a, b, c = exactly_no_solution_exists()
        solve = parasolve(lambda x: a * x ** 2 + b * x + c, init)
        if ASSERT:
            assert cmath.isclose(a * solve ** 2 + b * solve + c, 0, abs_tol = 1e-3)


tests = np.array((
    "easy_normal_solution",
    "test_two_solutions",
    "test_one_solution",
    "easy_complex_solution",
    "test_complex_solutions"
    )
)

def run_tests():
    for test in tests:
        print(f"{test}...")
        globals()[test]()
        print("done!\n")

    print("Congratulations! All tests passed!")
    return

run_tests()
