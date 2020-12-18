import matplotlib.pyplot as plt

def plot(textfile):
    avg = []
    fmax = []
    n = []
    i = 1
    with open(textfile) as file:
        lines = file.readlines()
        for line in lines:
            n.append(i)
            fmax.append(float(line.strip().split(";")[2].replace("[", "").replace("]","")))
            avg.append(float(line.strip().split(";")[3].replace("[", "").replace("]","")))
            i+=1
    return n, avg, fmax

    

if __name__ == "__main__":
    # seeds = [12345678, 98765431, 75356, 1546, 888888, 252554, 123876]
    # for seed in seeds:
    #     n, avg, fmax = plot(f"123/f1_{seed}_200_1.3_160_2.txt")
    #     plt.plot(n, avg, label=f"{seed}")
    # plt.show()
    func = "f1"
    mut = "1.3"

    for elite in range(10, 200, 10):
        n, avg, fmax = plot(f"{func}_12345678_200_{mut}_{elite}_2.txt")
        plt.plot(n, avg, label=f"{elite}")
    plt.legend(loc='lower right')
    plt.suptitle("Zależność średniej wartości pokolenia od numeru pokolenia")
    plt.ylabel("Średnia wartość")
    plt.xlabel("Numer pokolenia")
    plt.show()


    for elite in range(10, 200, 10):
        n, avg, fmax = plot(f"{func}_12345678_200_{mut}_{elite}_2.txt")
        plt.plot(n, fmax, label=f"{elite}")
    plt.legend(loc='lower right')
    plt.suptitle("Zależność maksymalnej wartości pokolenia od numeru pokolenia")
    plt.ylabel("Maksymalna wartość")
    plt.xlabel("Numer pokolenia")
    plt.show()

    # plt.legend(loc='lower right')
    # plt.suptitle("Zależność średniej wartości pokolenia od numeru pokolenia")
    # plt.ylabel("Średnia wartość")
    # plt.xlabel("Numer pokolenia")
    # plt.show()

    # plt.plot(n, fmax1, label="1546")
    # plt.plot(n, fmax2, label="75356")
    # plt.plot(n, fmax3, label="123876")
    # plt.plot(n, fmax4, label="252554")
    # plt.plot(n, fmax5, label="888888")
    # plt.plot(n, fmax6, label="98765431")
    # plt.plot(n, fmax7, label="12345678")
    # plt.legend(loc='lower right')
    # plt.suptitle("Zależność maksymalnej wartości pokolenia od numeru pokolenia")
    # plt.ylabel("Maksymalna wartość")
    # plt.xlabel("Numer pokolenia")
    # plt.show()