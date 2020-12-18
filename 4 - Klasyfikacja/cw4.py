import numpy as np
from random import shuffle
from scipy.optimize import minimize
from functools import partial
import matplotlib.pyplot as plt
# -------------------------------
# Data management
# -------------------------------


def read_from_file(filename):
    newlines = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            atributes = line.strip().split(",")[:4]
            atributes = [float(x) for x in atributes]
            name = line.strip().split(",")[4]
            newlines.append([name_to_numeric(name), atributes])
    return newlines


def name_to_numeric(name):
    if name == "Iris-setosa":
        return 0
    elif name == 'Iris-versicolor':
        return 1
    elif name == 'Iris-virginica':
        return 2


def split_data(data, ratio_a, ratio_b, ratio_c):
    if (ratio_a + ratio_b + ratio_c == 1):
        shuffle(data)
        length = len(data)
        sep_ab = int(ratio_a * length)
        sep_bc = sep_ab + int(ratio_b * length)

        test = data[:sep_ab]
        learn = data[sep_ab:sep_bc]
        valid = data[sep_bc:]
        return learn, test, valid
    else:
        raise Exception("Invalid ratio")


def group(data, no_classes):
    classes = [[] for n in range(0, no_classes)]
    for line in data:
        classes[line[0]].append(line)
    return classes


def make_set(data, given_index):
    new_data = []
    for index, atributes in data:
        if index == given_index:
            new_data.append([atributes[0], atributes[1], atributes[2], atributes[3], 1])
        else:
            new_data.append([atributes[0], atributes[1], atributes[2], atributes[3], -1])
    return np.array(new_data)
# -------------------------------
# Main algorithm
# -------------------------------


def get_lambda(_trainset, _validset):
    _lambda = [0.001, 0.01, 0.1, 0.5, 1, 100, 1000]
    _lambda_results = []
    for l in _lambda:
        _lambda_results.append(sum(make_test(_trainset, _validset, l)[0]))
    return sorted(zip(_lambda, _lambda_results), key=lambda x: x[1])[0][0]


def make_test(_trainset, _testset, _lambda):
    groups = group(_trainset, 3)

    fun01 = SVM(make_set(groups[0] + groups[1], 0), _lambda)
    fun12 = SVM(make_set(groups[1] + groups[2], 1), _lambda)
    fun20 = SVM(make_set(groups[2] + groups[0], 2), _lambda)

    errors = [0, 0, 0]
    lentab = [0, 0, 0]
    for value, atributes in _testset:
        if (check_vector(atributes, fun01, fun12, fun20) != value):
            errors[value] += 1
        lentab[value] += 1
    return errors, lentab


def SVM(_set, _lambda):
    attributes = _set[:, :-1]
    values = _set[:, -1]
    lamb = _lambda

    better_cost = partial(cost, _attr=attributes, _val=values, _lamb=lamb)
    result = minimize(better_cost, np.random.random(5)).x
    return result
# -------------------------------
# Testing
# -------------------------------


def check_vector(w, func0, func1, func2):
    status = [decision_function(np.matmul(func0[:4], w) - func0[4]), decision_function(np.matmul(func1[:4], w) - func1[4]), decision_function(np.matmul(func2[:4], w) - func2[4])]
    if (status[2] == -1) and (status[0] == 1):
        return 0
    if (status[0] == -1) and (status[1] == 1):
        return 1
    if (status[1] == -1) and (status[2] == 1):
        return 2
# -------------------------------
# Mathematic functions
# -------------------------------


def cost(w, _attr, _val, _lamb):
    better_ksi = partial(ksi, _attr=_attr, _val=_val)
    return np.linalg.norm(w, 2) * _lamb + better_ksi(w).sum()


def ksi(w, _attr, _val):
    return np.maximum(0, 1 - ((np.matmul(_attr, w[:4]) - w[4]) * _val))


def decision_function(x):
    if (x > 0):
        return 1
    else:
        return -1
# -------------------------------


if __name__ == "__main__":
    data = read_from_file("iris.data")
    classes = group(data, 3)
    
    for clas in classes:
        x = []
        y = []
        for value, arguments in clas:
            x.append(arguments[3])
            y.append(arguments[2])
        plt.plot(x, y, 'o')
    plt.legend(["Iris-setosa", 'Iris-versicolor', 'Iris-virginica'])
    plt.xlabel("Petal width")
    plt.ylabel("Petal length")
    plt.show()

    for clas in classes:
        x = []
        y = []
        for value, arguments in clas:
            x.append(arguments[1])
            y.append(arguments[0])
        plt.plot(x, y, 'o')
    plt.legend(["Iris-setosa", 'Iris-versicolor', 'Iris-virginica'])
    plt.xlabel("Sepal width")
    plt.ylabel("Sepal length")
    plt.show()

    for n in range(10):
        learn, test, valid = split_data(data, 0.6, 0.2, 0.2)
        lamb = get_lambda(test, valid)
        err, size = make_test(learn, test, lamb)
        print(f"Test {n}")
        print(f"Lambda {lamb}")
        print(f"Iris-setosa: {((size[0] - err[0])/size[0])*100:.2f}%")
        print(f"Iris-versicolor: {((size[1] - err[1])/size[1])*100:.2f}%")
        print(f"Iris-virginica: {((size[2] - err[2])/size[2])*100:.2f}%")
        print(f"Total accuracy: {(sum(size)-sum(err))/sum(size)*100:.2f}%\n")
