import numpy as np

### serdecznie polecam poniższą wspaniałą wizualizację
### arytmetyki zmiennoprzecinkowej:
### https://bartaz.github.io/ieee754-visualization/

# jaka jest największa liczba rzeczywista,
# dobrze reprezentowana w komputerze?
realmax = 2**(2**11-2-1023)*(2-1/2**52)
realmax * 2 # a to już nieskończoność
realmax * 2 / 2 # o dziwo (choć tak naprawdę nie o dziwo) to jest równe inf, a nie realmax
realmax + 1000 # ... = realmax
realmax + 1000 == realmax # True

# kilka naturalnych konwencji:
np.inf+np.inf # == np.inf
5/np.inf # == 0
5./0 # to Error
np.float64(5)/0 # ... a to już inf z ostrzeżeniem
np.inf/np.inf # to NaN
np.inf-np.inf # ... tak samo, jak to

### a co z najmniejszą liczbą dodatnią? Może coś takiego?
realmin = 2**(2**0-1023)*(1)
realmin

### ... nie do końca...
realmin / 2 == realmin

### .. realmin to najmniejsza "normalna", a są jeszcze subnormalne:

submin = 2**(1-1023)*2**(-52)
submin
print("{:.65g}".format(submin))
print(submin/2)
print(submin*2/3==submin)
print(submin/3*2)

### inny sposób dostania się do najmniejszej liczby:
y = 1
while y/2>0: # \label{(*)}
    y = y/2

y==submin

###  mogłoby się wydawać, że jeśli w (*) zastąpimy warunek
### y/2>0 przez y/2+1>1, to nic się nie zmieni
### ... ale w arytmetyce fl się zmienia!
y = 1
while 1+y/2>1:
    y = y/2

# teraz y to tzw. "epsilon maszynowy", czyli najmniejsza reprezentowalna
# liczba większa od 1
# w arytmetyce float64 to 2**{-52}


### Teraz piszemy programik, który oblicza
### pierwastki równania ax^2+bx+c w zależności
### od a,b,c
def solvq(a,b,c):
    delta=b**2-4*a*c
    if delta>0:
        x1 = (-b-np.sqrt(delta))/(2*a)
        x2 = (-b+np.sqrt(delta))/(2*a)
        print(f"Dwa pierwiastki: {x1} i {x2}")
        return x1,x2
    elif delta==0:
        x = -b/(2*a)
        print(f"Jeden pierwiastek: {x}")
        return x
    else:
        print("Brak pierwiastków")
        return np.nan

### testujemy
solvq(1,-2,1)
solvq(1,0,-1)
solvq(1,0,1)
solvq(1,6,5)

### niby wszystko ok... ale weźmy
solvq(1,-2*np.sqrt(3),3)

### i się psuje!
# a psuje się dlatego, że niby delta powinna
# być równa zero, ale taka nie wychodzi.

# podobne tego typu obserwacje
.1+.2==.3
(4/3-1)*3
10 + .1 == 10.1
10.1 - 10 == .1

### morał: do sprawdzania równości
### w arytmetyce fl warto rozważyć
### coś w stylu np.isclose()


### to teraz napiszamy funkcję obliczającą
### przybliżenie pochodnej cosinusa:
### (i od razu pokazującą błąd bezwzględny)
def cosprim(x,eps):
    dcos = (np.cos(x+eps)-np.cos(x))/eps
    return dcos, abs(dcos+np.sin(x))

## testujemy
cosprim(1, .1) # ujdzie
cosprim(1, .01) # nieźle
cosprim(1, 1e-30) # fatalnie! (1e-30 to naukowy zapis 10**(-30))

## wiadomo dlaczego fatalnie, 1e-30 jest mniejsze od epsilona maszynowego
## ale nawet dla epsilona maszynowego nie jest specjalnie dobrze
cosprim(1, 2**(-52))

# obejrzyjmy błąd na wykresie
import matplotlib.pyplot as plt
cosprim(1,1/2**60)
powers = np.arange(1,60)
powers
dcos, err = cosprim(1,1/2**powers)
plt.plot(powers,err)
plt.yscale('log')

## najmniejszy błąd dla h około 2**25, czyli z grubsza pierwiastek z epsilona
## na labach wytłumaczono (z grubsza), dlaczego


### na zakończenie funkcja licząca e^x za pośrednictem szeregu
### podobnie jak poprzednio patrzymy na błąd i na błąd względny
def truncexp(x,N):
    res = 1
    w = 1
    for k in range(1,N+1):
        w *= x/k
        res += w
    err = abs(res-np.exp(x))
    return res, err, err/np.exp(x)

# dla dodatnich x dziala nieźle
truncexp(1,50)
truncexp(30,500)

# ale dla ujemnych beznadziejnie
truncexp(-30,500)

## dlaczego?
## redukcja cyfr znaczących przy odejmowaniu!

## akurat tutaj, by to sprytnie ominąć
## moglibyśmy zwracać 1/trncexp(-x,N) dla x<0


# dodatek: trochę lepsze liczenie pochodnej:
def cosprim2(x,eps):
    dcos = (np.cos(x+eps)-np.cos(x-eps))/(2*eps)
    return dcos, abs(dcos+np.sin(x))

cosprim(1,1/2**60)
powers = np.arange(1,60)
powers
dcos, err = cosprim2(1,1/2**powers)
plt.plot(powers,err)
plt.yscale('log')
