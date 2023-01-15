from matplotlib import pyplot
from matplotlib import image
import numpy as np

### celem zajęć jest połączenie poniższych dwóch rysunków
### przy pomocy punktów kontrolnych wskazanych przez użytkownika
im1 = image.imread("img1.jpg")
im2 = image.imread("img2.jpg")

### ... na razie nie widać, jakie to rysunki...
### te im1, im2 to ndarraye z 3ma wymiarami,
### kodujące wartości w kanałach RGB
type(im1)
np.shape(im2)

### możemy je pooglądać:
pyplot.imshow(im1)

### w przyszłości będziemy chcieli pobierać współrzędne
### punktów z tych rysunków. Poniższe polecenie pozwala
### klikać w rysunek tyle razy, ile wynosi argument
### i pobiera współrzędne punktów
pyplot.ginput(3)

### na początku jednak chcemy połączyć rysunki
### robimy to poniżej
n1, m1, _ = np.shape(im1)
n2, m2, _ = np.shape(im2)

d = n2 - n1
im1_resized = np.vstack( (im1, np.zeros((d, m1, 3), dtype=int)) )
im_full = np.hstack( (im1_resized, im2) )
pyplot.imshow(im_full)

### teraz chcemy mieć kod, który
### "prawy rysunek" (ten zniekształcony)
### przekształci  przy użyciu zadanej macierzy A i wektora B

def im_modify(im1, im2, A, b):
    n1, m1, _ = np.shape(im1)
    n2, m2, _ = np.shape(im2)

    d = n2 - n1
    im1_resized = np.vstack( (im1, np.zeros((d, m1, 3), dtype=int)) )
    nowa_plansza = np.hstack( (im1_resized, np.zeros_like(im2)) )
    for i in range(n2):
        for j in range (m2):
            k, l = np.round( A@np.array([i,j+m1]) + b ).astype(int)
            if m1 <= l < m1+m2 and 0 <= k < n2:
                nowa_plansza[k, l] = im2[i, j]

    return nowa_plansza

A = np.array([[1,0],[0,1]])
b = np.array([50,0])
nowa_plansza = im_modify(im1, im2, A, b)
pyplot.imshow(nowa_plansza)

### Pobieramy parzyście wiele punktów
### w parach "punkt z prawego, odpowiadający punkt z lewego"
pyplot.imshow(im_full)
x = pyplot.ginput(8)

### Teraz musimy znaleźć odpowiednie A i b
### Jest to sześć niewiadomych (4 na A i 2 n a b),
### na które możemy nałożyć kwadraową karę związaną
### z tym jak dobrze jedne punkty przechodzą na drugie
### Po przyjrzeniu się (zrobione na tablicy) okazuje się, że
### jest to LZNK
x = np.array(x)
x = x[:,(1,0)]
P = x[::2]
P = np.hstack((P, np.ones( (np.shape(P)[0],1) ))) # co drugi wiersz
Q = np.linalg.lstsq(P, x[1::2])[0]
A = Q[0:2,0:2].T
b = Q[2,:]

### co wyszło?
nowa_plansza = im_modify(im1, im2, A, b)
pyplot.imshow(nowa_plansza)
