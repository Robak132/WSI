import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy


def generate_herd(n_pop, center, dev):
    a = np.random.normal(center, dev, n_pop)
    b = np.random.normal(center, dev, n_pop)
    return [np.array([[a], [b]]) for a, b in zip(a, b)]


def mutate(x, mutdist):
    x[0] += np.random.normal(0, mutdist)
    x[1] += np.random.normal(0, mutdist)
    return x


def get_score(x, func):
    score = []
    for elem in x:
        score.append((elem, func(elem)))
    score.sort(key=lambda x: x[1], reverse=True)
    keys, values = zip(*score)
    avr = np.average(values)
    return list(keys), list(values), avr


def evolve(base_pop, bugdet, pop, mutdist, elite_pop, match_size, func, seed, fil=100):      
    np.random.seed(seed)
    totalavr = []
    totalf0 = []
    data = []
    print(f"{seed};pop={pop};mutdist={mutdist};elite_pop={elite_pop};match_size={match_size}")

    """ Tworzenie populacji początkowej """
    herd = generate_herd(pop, base_pop[0], base_pop[1])
    herd, herd_score, avr = get_score(herd, func)

    for npop in range(bugdet // pop - 1):
        totalavr.append(avr)
        totalf0.append(herd_score[0])
        data.append([npop, herd[0], herd_score[0],avr])
        if npop % fil == 0:
            print(f"{npop:<10}{str(herd[0].tolist()):<48}f(x)={str(herd_score[0]):<25}avr={avr}")

        """ Selekcja """
        temp_herd = []
        for i in range(pop - elite_pop):
            match = []
            for m in range(match_size):
                index = np.random.randint(0, pop)
                match.append((herd[index], herd_score[index]))
            match.sort(key=lambda x: x[1], reverse=True)
            best = deepcopy(match[0][0])
            temp_herd.append(best)

        """ Mutacja """
        for elem in temp_herd:
            elem = mutate(elem, mutdist)

        """ Sukcesja """
        new_herd = herd[:elite_pop] + temp_herd
        herd = new_herd
        herd, herd_score, avr = get_score(herd, func)

    """ Zapis po ostatnim ocenieniu"""
    totalavr.append(avr)
    totalf0.append(herd_score[0])

    """ Zapis do pliku """
    with open(f"{func.__name__}_{seed}_{pop}_{mutdist}_{elite_pop}_{match_size}.txt", "w+") as file:
        for pop, bestobj, fobj, avr in data:
            file.write(f"{pop};{bestobj.tolist()};{fobj};{avr}\n")


def fi(x, u, E):
    return (math.exp(-0.5 * np.matmul(np.matmul(np.transpose(x - u), np.linalg.inv(E)), x - u))) / (math.sqrt((2 * math.pi) ** (np.size(x)) * np.linalg.det(E)))


def f1(x):
    return (
        fi(x, np.array([[14], [-11]]), np.array([[1.3, -0.5], [-0.5, 0.8]]))
        + fi(x, np.array([[10], [-10]]), np.array([[1.7, 0.4], [0.4, 1.2]]))
        + fi(x, np.array([[7], [-13]]), np.array([[1, 0], [0, 1.5]]))
    )


def f2(x):
    return -(-20 * np.exp(-0.2 * np.sqrt(0.5 * np.matmul(np.transpose(x), x)))
        - np.exp(0.5 * (np.cos(2 * math.pi * x[0]) + np.cos(2 * math.pi * x[1]))) + math.e + 20)


if __name__ == "__main__":
    seeds = [12345678, 98765431, 75356, 1546, 888888, 252554, 123876]
    
    evolve([0, 1], 1000000, 200, 1.3, 160, 2, f1, seeds[0])

    # for seed in seeds:   
        # evolve([0, 1], 1000000, 200, 1.3, 160, 2, f1, seed)
    
    # seed = seeds[0]
    # for elite in range(10, 100, 10):   
    #    evolve([0, 1], 1000000, 200, 1.3, elite, 2, f1, seed)


    # for seed in seeds:   
        # evolve([3, 1], 1000000, 200, 0.5, 160, 2, f2, seed)
    
    # seed = seeds[0]
    # for elite in range(10, 100, 10):   
    #    evolve([3, 1], 1000000, 200, 0.5, elite, 2, f2, seed)


