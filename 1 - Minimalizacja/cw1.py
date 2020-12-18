"""
Jakub Robaczewski
Przeszukiwanie przestrzeni
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import timeit


class NewtonMethod:
    def __init__(self, dim=2):
        self.dim = dim

    def newx(self, x, b):
        d = np.matmul(np.linalg.inv(self.hf(x)), self.grad(x))
        return x - d * b

    def generate(self, dim=2):
        """
        Generuje macierz (wektor) o podanym wymiarze.
        """
        rand = []
        for i in range(dim):
            rand.append([random.randrange(0, 10) + random.randrange(0, 100) / 100])
        return np.array(rand)

    def test_B(self, x=None, minb=0.1, maxb=2, stepb=0.1, stop_acc=0.000001, stop_n=1000):
        """
        Testuje algorytm dla wskazanych b
        Zwraca listę kroków algorytmu, listę czasów, listę wyników i listę wektorów końcowych.
        """
        if x is None:
            x = self.generate(self.dim)
        nl = []
        res = []
        wect = []
        time = []

        with open(f"{x.tolist()}.txt", 'w+') as file:
            for b in np.arange(minb, maxb, stepb):
                tc = 0
                for i in range(5):  # Średnia wartość czasu
                    t = timeit.default_timer()
                    n, w = self.run(b, x, stop_acc, stop_n)
                    t = timeit.default_timer() - t
                    tc += t
                t = tc / 5
                r = float(self.f(w))

                file.write(f"{b};{n};{t};{r};{w.tolist()}\n")

                time.append(t)
                nl.append(n)
                res.append(r)
                wect.append(w)

        return nl, time, res, wect

    def run(self, b, x=None, stop_acc=0.000001, stop_n=1000, debug=False, plot=False):
        """
        Wykonuje algorytm zaczynając w punkcie (wektorze) x, o danym wzmocnieniu b.
        bool debug - kontroluje wypisywanie informacji o każdym kroku
        Zwraca liczbę kroków n, otrzymany wektor końcowy w.
        """
        if x is None:
            x = self.generate(self.dim)
        stop = False
        n = 0

        hist_x = []
        hist_f = []

        while not stop:
            hist_x.append(x)
            hist_f.append(self.f(x))
            if debug:
                print(f"{n:>4}: {x.tolist()};{self.f(x)};{(self.hf(x)).tolist()};{np.linalg.norm(x-hist_x[-1])}")
            x = self.newx(x, b)
            n += 1
            if (abs(np.linalg.norm(x - hist_x[-1])) < stop_acc) or n == stop_n:
                stop = True
        if debug:
            print(f"{n:>4}: {x.tolist()};{self.f(x)};{(self.hf(x)).tolist()};{np.linalg.norm(x-hist_x[-1])}")

        return n, x


class NewtonMethod_A (NewtonMethod):
    def f(self, x):
        return -1 * np.matmul(np.transpose(x), x)

    def grad(self, x):
        return -2 * x

    def hf(self, x):
        return -2 * np.identity(self.dim)


class NewtonMethod_B (NewtonMethod):
    def f(self, x):
        return -1 * np.matmul(np.transpose(x), x) + 1.1 * np.cos(np.matmul(np.transpose(x), x))

    def grad(self, x):
        return -2 * x - 2.2 * x * np.sin(np.matmul(np.transpose(x), x))

    def hf(self, x):
        return -2 * np.identity(self.dim) - 2.2 * np.identity(self.dim) * np.sin(np.matmul(np.transpose(x), x)) - 4.4 * np.matmul(x, np.transpose(x)) * np.cos(np.matmul(np.transpose(x), x))


def make_plot(x, y1, y2=None, y3=None, y4=None, title=None, xtitle=None, ytitle=None, label=None, legend=None, filename=None):
    plt.plot(x, y1)
    if y2 is not None:
        plt.plot(x, y2)
    if y3 is not None:
        plt.plot(x, y3)
    if y4 is not None:
        plt.plot(x, y4)

    if title is not None:
        plt.suptitle(title)
    if xtitle is not None:
        plt.xlabel(xtitle)
    if ytitle is not None:
        plt.ylabel(ytitle)
    if legend is not None:
        plt.legend(legend, loc='upper left')
    if label is not None:
        for index in range(len(label)):
            plt.annotate(f"{label[index]:0.2f}", (x[index], y1[index]))

    plt.xlim(x[0], x[-1])
    plt.grid()

    if filename is not None:
        plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # N = NewtonMethod_A(2)
    # w1 = N.test_B(x=np.array([[-1], [-1]]), minb=0.1, maxb=2, stepb=0.1)
    # w2 = N.test_B(x=np.array([[2], [2]]), minb=0.1, maxb=2, stepb=0.1)
    # w3 = N.test_B(x=np.array([[3], [3]]), minb=0.1, maxb=2, stepb=0.1)

    # data = 0
    # make_plot(np.arange(0.1, 2, 0.1), w1[data], w2[data], w3[data], title="Zależność czasu działania algorytmu od wartości B i punktu początkowego", xtitle="Parametr B", ytitle="Czas [s]", legend=["[-1], [-1]", "[2], [2]", "[3], [3]"], filename="Fig1.png")
    # make_plot(np.arange(0.1, 2, 0.1), w1[data], w2[data], w3[data], title="Zależność liczby kroków algorytmu od wartości B i puntku początkowego", xtitle="Parametr B", ytitle="Kroki", legend=["[-1], [-1]", "[2], [2]", "[3], [3]"], filename="Fig1.png")
    # N.run(2, x=np.array([[-1], [-1]]), plot=True)

    # N = NewtonMethod_B(2)
    # w1 = N.test_B(x=np.array([[0.5], [0.5]]), minb=0.1, maxb=2, stepb=0.1)
    # w2 = N.test_B(x=np.array([[0.95], [0.95]]), minb=0.1, maxb=2, stepb=0.1)
    # w3 = N.test_B(x=np.array([[1], [1]]), minb=0.01, maxb=2, stepb=0.01)
    # w4 = N.test_B(x=np.array([[3], [3]]), minb=0.1, maxb=2, stepb=0.1)

    # make_plot(np.arange(0.1, 2, 0.1), w1[2], w2[2], w3[2], w4[2], title="Zależność f(x) od wartości B i punktu początkowego", legend=["[0.5], [0.5]", "[0.95], [0.95]", "[-1], [-1]", "[3], [3]"], xtitle="Parametr B", ytitle="F(X)", filename="Fig2.png")
    # make_plot(np.arange(0.1, 2, 0.1), w1[0], w2[0], w3[0], w4[0], title="Zależność liczby kroków algorytmu od wartości B i puntku początkowego", legend=["[0.5], [0.5]", "[0.95], [0.95]", "[-1], [-1]", "[3], [3]"], xtitle="Parametr B", ytitle="Kroki", filename="Fig2.png")
    # make_plot(np.arange(0.1, 2, 0.1), w1[1], w2[1], w3[1], w4[1], title="Zależność czasu działania algorytmu od wartości B i punktu początkowego", legend=["[0.5], [0.5]", "[0.95], [0.95]", "[-1], [-1]", "[3], [3]"], xtitle="Parametr B", ytitle="Czas [s]", filename="Fig2.png")
    # N.run(1.2, x=np.array([[0.95], [0.95]]), plot=True)
