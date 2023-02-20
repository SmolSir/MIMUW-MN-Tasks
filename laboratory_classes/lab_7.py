import numpy as np
def vanilla_trapez_int(f, a, b, k):
    N = 2**k
    x = np.linspace(a, b, N+1)
    h = (b-a) / N
    return h*( f(x[0]) + f(x[N]) )/2 + h*np.sum( f(x[1:N]))

def trapez_int(f, a, b, k):
    h = b-a
    Q = (f(a)+f(b))*h/2
    hist = [Q]
    for i in range(1,k):
        h = h/2
        x_new = np.linspace(a+h,b-h, 2**(i-1))
        Q = Q/2 + h*np.sum( f(x_new) )
        hist.append(Q)
    return np.array(hist)

def simson_int(f, a, b, k):
    h = b-a
    c = (b+a)/2
    Q = (f(a)+f(b)+4*f(c))*h/6
    srodki = np.array([c])
    hist = [Q]
    for i in range(1,k):
        h = h/2
        nowe_srodki = np.linspace(a+h/2, b-h/2, 2**i)
        Q = Q/2 - h/6 * 2*np.sum(f(srodki)) + 4/6*np.sum(f(nowe_srodki))
        hist.append(Q)
        srodki = nowe_srodki
    return np.array(hist)

f = lambda x: x**2

np.abs(1/3 - trapez_int(f, 0, 1, 5))
np.abs(1/3 - simson_int(f, 0, 1, 5))
