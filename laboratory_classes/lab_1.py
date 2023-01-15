### Skrypt z labów z MN, 2022-10-05

# zaczęliśmy od podstawowych operacji arytmetycznych
# możliwych do wykonania w Pythonie:
1+1
23*45
3**8 # potęgowanie
3^8 # ... nie mylić z xorowaniem

3/8, 4/2, type(4), type(4/2) # przy dzieleniu trzeba czasem uważać na typy
# jeśli chcemy dzielić "inty", można wykorzystać operator //
# ... ale nas to raczej nie będzie interesować

# do bardziej wyszukanych operacji artytmetycznych
# można zaimportować moduł math
import math
math.sqrt(5)
math.sin(3)

# a do numeryki (co nas interesuje) moduł numpy
import numpy as np
v = np.array([1,3,2,5]) # definiowanie wektora
w = np.array([3,9,1,2])
v+w; v*w # dodawanie/mnozenie po wspolrzednych
sum(v*w); np.dot(v,w); v@w # trzy sposoby na iloczyn skalatny

# macierze i operacje z nimi związane:
A = np.array([[1,2,4],[3,4,1],[1,1,1]])
v = np.array([1,3,5])
np.dot(A,v)
A.dot(v)
A@v
np.size(A);np.shape(A)
np.shape(v)
A[0:2,0:2]
np.shape(A[0:1,])
np.shape(A[0,])
A.T

np.linalg.det(A)
# co ciekawe, wyznacznik nie wyszedł całkowity, a powinien.
# kwestia tego, że det jest liczony poprzez rozkład LU (jak?)
# a w związku z tym mamy błędy przybliżeń naszej arytmetyki

# w ramach ćwiczenia, funkcja licząca wyznacznik
# poprzez rozwinięcie Laplace'a
def laplace_det(A):
    n, m = A.shape # można sprytnie n=len(A), (c)SŚ
    if n != m:
        raise np.linalg.LinAlgError("Last 2 dimensions of the array must be square")
    if n == 1:
        return A[0, 0]
    determinant = 0
    for i in range(n):
        # np.r_ odpowiada za "łączenie slice'ów"
        determinant += (-1) ** i * A[i, 0] * laplace_det(A[np.r_[0:i, (i + 1):n], 1:n])
    return determinant

# uklad
# .1036x + .2122y = .7381
# .2081x + .4247y = .9327

# rozwiązujemy zaokrąglając do trzech miejsc po przecinku
a = .208/.104
a = np.round(a,3)
round(.933 - a*.738, 3) / round(.425 - a*.212, 3) # y=-543

# rozwiązujemy zaokrąglając do czterech miejsc po przecinku
a = .2081/.1036
a = round(a,4)
round(.9327 - a*.7381, 4) / round(.4247 - a*.2122, 4) # y = 366.6

# rozwiązujemy tak dobrze, jak potrafimy
a = .2081/.1036
(.9327 - a*.7381) / (.4247 - a*.2122) # y = 356.6

# a używając numpy można napisać:
A = np.array([[.1036, .2122], [.2081, .4247]])
b = np.array([.7381, .9327])

np.linalg.solve(A,b) # rozwiązanie Ax=y
np.linalg.inv(A)@b # ZŁO! Kiedy tylko możemy tego uniknąć, unikamy inv!


# macierz Hilberta 3x3
H = np.array([[1,1/2,1/3],[1/2,1/3,1/4],[1/3,1/4,1/5]])
v = np.array([1,1,1])
b = H@v
np.linalg.solve(H,b) # wychodzi [1,1,1], tak jak powinno

# to teraz bierzemy macierz Hilberta 20x20
n = 20
H = np.array([[1/i for i in range(j,n+j)] for j in range(1,n+1)])
v = np.ones(n)
b = H@v
np.linalg.solve(H,b) # koszmar! zupełnie co innego, niż wektor jedynek

# spróbujmy obliczyć teraz wartości całek
# a_n = int_0^1 x^n/(x+5); a_n=1/n-5a_{n-1}

# wiadomo, że a_0=ln(6/5), ponadto można udowodnić, że
# a_n=1/n - 5a_{n-1}
# wykorzystajmy to!

n = 10
a = np.log(6/5)
for i in range(1,10):
    a = 1/i - 5*a

# dla n=10 wygląda OK,
# ale dla n=30 koszmar - wychodzą ogromne liczby,
# a cały czas powinny leżeć w przedziale [0,1]

# na koniec, chałupnicza metoda liczenia symbolu Newtona
def my_fact(n):
    """Moja autorska silnia"""
    return np.prod(np.arange(1,n+1))

def my_binom(n, k):
    """Mój autorski symbol Newtona"""
    return my_fact(n) / ( my_fact(k)*my_fact(n-k) )

my_binom(100,4) # wykrzacza się!
# ... i nic dziwnego - silnie ekspodują błyskawicznie

# można w tej sytuacji albo skrócić (n-k)!
# i policzyć iloczyn n/1 * (n-1)/2 * (n-2)/3 * ... * (n-k+1)/k
def my_binom2(n, k):
    k = min(k, n-k)
    return np.prod(np.arange(n,n-k,-1) / np.arange(1,k+1))

# albo po prostu wykorzystać trójkąt Pascala
def my_binom3(n,k):
    v = [1 for _ in range(n-k+1)]
    for _ in range(k):
        for i in range(1,n-k+1):
            v[i] += v[i-1]
    return v[-1]

# to ostatnie jest o tyle lepsze, że choć ma większy koszt
# (n*k vs (n-k)), to cały czas operuje na intach, więc
# właściwie nie ma ograniczenia na wynik

my_binom3(200,50)