import numpy as np
def bisect_solve(f, a, b, eps):
    m = (a+b)/2
    fm = f(m)
    hist = [m]
    while abs(fm) > eps:
        if fm*f(a) < 0:
            b = m
        else:
            a = m
        m = (a+b)/2
        fm = f(m)
        hist.append(m)
    return m, np.array(hist)

def fpm_solve(f, a, b, eps):
    fa, fb = f(a), f(b)
    m = (a*fb - b*fa)/(fb-fa)
    fm = f(m)
    hist = [m]
    while abs(fm) > eps:
        if fm*f(a) < 0:
            b = m
        else:
            a = m
        m = (a * fb - b * fa) / (fb - fa)
        fm = f(m)
        hist.append(m)
    return m, np.array(hist)

def section_solve(f, a, b, eps):
    fa, fb = f(a), f(b)
    m = (a*fb - b*fa)/(fb-fa)
    fm = f(m)
    hist = [m]
    while abs(fm) > eps:
        a, b, fa, fb = b, m, fb, fm
        m = (a * fb - b * fa) / (fb - fa)
        fm = f(m)
        hist.append(m)
    return m, np.array(hist)

def newton_solve(f, df, a, eps):
    fa = f(a)
    hist = [a]
    while abs(fa) > eps:
        a = a - fa / df(a)
        fa = f(a)
        hist.append(a)
    return a, np.array(hist)

def numerical_newton_solve(f, a, h, eps):
    fa = f(a)
    hist = [a]
    while abs(fa) > eps:
        dfa = (f(a+h)-fa)/h
        a = a - fa / dfa
        fa = f(a)
        hist.append(a)
    return a, np.array(hist)

def steffensen_solve(f, a, eps):
    fa = f(a)
    hist = [a]
    while abs(fa) > eps:
        h = min(abs(fa), 1e-3)
        dfa = (f(a+h)-fa)/h
        a = a - fa / dfa
        fa = f(a)
        hist.append(a)
    return a, np.array(hist)

def halley_solve(f, df, d2f, a, eps):
    fa = f(a)
    hist = [a]
    while abs(fa) > eps:
        dfa = df(a)
        a = a - fa / (dfa - fa/dfa*d2f(a)/2 )
        fa = f(a)
        hist.append(a)
    return a, np.array(hist)

def multidim_newton_solve(f, df, a, eps):
    fa = f(a)
    hist = [a]
    while np.linalg.norm(fa) > eps:
        # a = a - np.linalg.inv(df(a))@fa # PASKUDNIE!
        a = a - np.linalg.solve(df(a), fa)
        fa = f(a)
        hist.append(a)
    return a, np.array(hist)

f = lambda x: x**2-9

_, h1 = bisect_solve(f, 0, 5, 1e-6)
np.abs(h1-3)

_, h1 = fpm_solve(f, 0, 5, 1e-6)
np.abs(h1-3)

_, h1 = section_solve(f, 0, 5, 1e-16)
np.abs(h1-3)

df = lambda x: 2*x
_, h1 = newton_solve(f, df, 6, 1e-16)
np.abs(h1-3)

_, h1 = numerical_newton_solve(f, 6, 1e-4, 1e-16)
np.abs(h1-3)

_, h1 = steffensen_solve(f, 6, 1e-16)
np.abs(h1-3)

f = lambda x: x**3-27
df = lambda x: 3*x**2
d2f = lambda x: 6*x
_, h1 = halley_solve(f, df, d2f, 6, 1e-16)
np.abs(h1-3)

f = lambda x: x**3
df = lambda x: 3*x**2
_, h1 = newton_solve(f, df, 6, 1e-16)
np.abs(h1)

_, h1 = halley_solve(f, df, d2f, 6, 1e-16)
np.abs(h1)

f = lambda x: np.array([x[0]**2-x[1]**2, x[0]+x[1]-4])
df = lambda x: np.array([ [2*x[0], -2*x[1]], [1,1] ])
x0 = np.array([3.,5.], dtype="float")
_, h1 = multidim_newton_solve(f, df, x0, 1e-16)
np.abs(h1)
df(x0)

