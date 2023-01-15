# Te zajęcia dotyczyły interpolacji wielomianowej/splajnowej
# i tego, jak przekłada się ona na interpolację w normie supremum

# poniżej moduły i funkcje, które będą wykorzystywane
import numpy as np
from scipy.interpolate import BarycentricInterpolator as BI # wielomian interpolacyjne
from scipy.interpolate import InterpolatedUnivariateSpline as Spline # splajn interpolacyjny
from matplotlib import pyplot as plt

## Funkcja pomocnicza do animacji
## (na zajęciach było bez funkcji, na bieżąco modyfikowaliśmy skrypt

def animateInterpolation(f, a, node_type, inter_type, Nmin, Nmax, t_stop=1, k=3):
    """Funkcja animuje wielomian/splajn interpolacyjny
    dla funkcji f na przedziale [-a,a]. Argumenty:
    -- f: funkcja
    -- a: liczba
    -- node_type: "eq" dla równoodległych, "cz" dla czebyszewa
    -- inter_type: "poly" dla wielomianów, "spline" dla splajnów
    -- Nmin, Nmax: ile węzłów
    -- k: rząd splajnu
    --t_stop: czas przerwy"""
    for n in range(Nmin, Nmax):
        plt.cla()
        if node_type == "cz":
            nodes = a*np.cos((2*np.arange(1,n+1)-1)*np.pi/(2*n))
        else:
            nodes = np.linspace(-a,a, num=n)
        if inter_type == "poly":
            p = BI(nodes, f(nodes))
        else:
            p = Spline(nodes, f(nodes),k=3)
        x = np.linspace(-a,a, num=1000)
        plt.plot(x,f(x))
        plt.plot(x,p(x))
        plt.scatter(nodes,f(nodes))
        plt.pause(t_stop)

### eksperymenty:
##### pod koniec jest już OK:
animateInterpolation(lambda x:np.sin(2*x), 5, "eq", "poly", 5, 20, .5, k=3)

##### na krańcach cały czas szaleje (przykład Rungego:
animateInterpolation(lambda x: 1/(x**2+1), 5, "eq", "poly", 5, 20, .5, k=3)

##### lepiej z węzłami Czebyszewa (dużo lepiej!)
animateInterpolation(lambda x: 1/(x**2+1), 5, "cz", "poly", 5, 18, .5, k=3)

##### ze splajnami też ok (nawet lepiej niż z Czebyszewem)
animateInterpolation(lambda x: 1/(x**2+1), 5, "eq", "spline", 5, 18, .5, k=3)

###################################################
# A teraz będziemy prowadzić krzywą przez punkty:
###################################################
N = 6
plt.cla()
x = np.random.rand(N)
y = np.random.rand(N)

for i, label in enumerate(np.arange(N)):
    plt.text(x[i], y[i],label)

## pierwszy pomysł jest taki, by w określonych
## punktach czasowych (t=1,2,...,N) zgadzały
## się współrzędne iksowe i igrekowe, które
## interpolujemy przy pomocy wielomianów/splajnów
t = np.arange(N).astype(float)
px = Spline(t,x,k=3)
py = Spline(t,y,k=3)
tgrid = np.linspace(0,t[-1],200)
plt.plot(px(tgrid),py(tgrid),color="orange")

## ale to zachowuje się chaotycznie!
## lepiej zrobić tak, by te punkty czasowe były
## proporcjonalne do długości odcinków między kolejnymi punktami
plt.cla()
for i, label in enumerate(np.arange(N)):
    plt.text(x[i], y[i],label)
t = np.append(0,np.cumsum(np.sqrt(np.diff(x)**2+np.diff(y)**2)))
px = Spline(t,x,k=3)
py = Spline(t,y,k=3)
tgrid = np.linspace(0,t[-1],200)
plt.plot(px(tgrid),py(tgrid),color="orange")
