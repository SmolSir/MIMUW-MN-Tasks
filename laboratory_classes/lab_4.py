# Na tych zajęciach symulowaliśmy graf stron internetowych
# i zaimplementowaliśmy algorytm PageRank,
# który stronie internetowej przypisuje (z grubsza) częstość
# z jaką bot losowo wędrujący po grafie odwiedzałby daną stronę

import numpy as np

## parametry (zaczęliśmy od niewielkich)
n = 5 # liczba stron
m = 2 # przeciętna liczba odnośników na stronie
alpha = .8 # szansa na teleportację na losową stronę

## 1. Konstruujemy (losową) macierz przejścia
p = m/n
A = np.random.rand(n,n)
B = (A < p).astype(float) # losowa macierz incydencji
out_degs = B.sum(1, keepdims=True) # stopnie zewnętrzne wierzchołków
dead_ends = (out_degs == 0).flatten() # indykator stron, z których nie ma odnośników ("martwych")
P = B.copy() # nasza macierz przejścia, bez teleportacji (jeszcze będziemy nad nią pracować)
P[~dead_ends,:] = P[~dead_ends,:] / out_degs[~dead_ends] # wartości macierzy przejścia dla "żywych" stron
P[dead_ends,:] = 1/n # wartości macierzy przejścia dla "martwych" stron
Q = alpha*P + (1-alpha)*np.ones_like(P)/n # finalna macierz przejścia z teleportacją

## 2. Szukamy wektora własnego (poprzez zwykłe rozwiązanie układu równań)
Qm = alpha*P - alpha*np.ones_like(P)/n
x = np.linalg.solve(np.identity(n)-Qm.T, np.ones(n))
### to by się wysypało dla względnie niewielkich wartości n, np. 1e3

## 3. Wykorzystujemy metodę potęgową
x = np.random.rand(n)
for _ in range(100):
    x = x@Q
    x = x/sum(abs(x))

### ... choć pewnie zamiast nie-wiadomo-ile-razy-w-pętli
### lepiej zrobić while

tol = 1e-5
x = np.random.rand(n)
z = np.zeros(n)
while np.linalg.norm(x-z)>tol:
    z = x
    x = x@Q
    x = x/sum(abs(x)) # Tak naprawdę to niepotrzebne, bo mnożymy przez macierz stochastyczną
x

### Prawda jest taka, że to się ciągle wysypuje dla względnie niewielkich
### wartości n

### Skorzystajmy z tego, że macierze, na których pracujemy, są rzadkie

## 4. Wykorzystujemy rzadkość macierzy
import scipy.sparse
A = scipy.sparse.rand(n, n, density=p)
B = (A > 0).astype(float)
out_degs = np.asarray(B.sum(1)).flatten()
dead_ends = np.array(out_degs == 0)
open_ind = (~dead_ends).nonzero()[0]
P = B.copy()
D = scipy.sparse.coo_matrix((1/out_degs[~dead_ends], (open_ind, open_ind)), shape=(n,n))
P = D.dot(P)
P[dead_ends,:] = 1/n

# Q = alpha*P + (1-alpha)*scipy.sparse()/n # to byłoby bez sensu, stracilibyśmy rzadkość!

x = np.random.rand(n)
z = np.zeros(n)
while np.linalg.norm(x - z) > tol:
    z = x
    # poniżej chytry sposób na wykonanie Q*x, bez tracenia rzadkości
    x = alpha*P.T.dot(x) + (1-alpha)*np.mean(x)*np.ones(n)
    x = x/sum(abs(x))


### check
Ps = P.toarray()
Q = alpha*Ps + (1-alpha)*np.ones_like(Ps)/n
x@Q

## check
Pi = B.copy()
Pi[~dead_ends,:] = Pi[~dead_ends,:] / out_degs.reshape(n,1)[~dead_ends]
Pi[dead_ends,:] = 1/n
Pi.toarray()
