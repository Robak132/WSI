import numpy as np
import statistics as st
from math import sqrt, pi, exp, inf
from random import shuffle
from itertools import combinations
import matplotlib.pyplot as plt


def read_from_file(filename):
    newlines = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            atributes = line.strip().split(",")[1:]
            atributes = [float(x) for x in atributes]
            name = int(line.strip().split(",")[0]) - 1
            newlines.append([name, atributes])
    return newlines


def divide(ratio, data):
    shuffle(data)
    size = int(len(data) * ratio)
    group_a = data[:size]
    group_b = data[size:]
    return group_a, group_b


def group(data, no_classes):
    classes = [[] for n in range(0, 3)]
    for line in data:
        classes[line[0]].append(line)
    return classes


def pick_parameters(_data, a, b):
    pickeddata = []
    for row in _data:
        pickeddata.append([row[0], row[1][a], row[1][b]])
    return pickeddata


def cross_validate(data, n, no_classes):
    block_size = int(len(data) / n)
    blocks = []

    best = 0
    best_params = []

    for i in range(n):
        blocks.append(data[i * block_size:(i + 1) * block_size])
    for param1, param2 in combinations(range(0, len(data[0][1])), 2):
        actotal_list = []
        for test in range(len(blocks)):
            train_set = []
            for block in blocks[0:test] + blocks[(test + 1):]:
                for line in block:
                    train_set.append(line)
            train_set = pick_parameters(train_set, param1, param2)
            test_set = pick_parameters(blocks[test], param1, param2)
            accuracy = make_test(test_set, train_set, no_classes)
            actotal_list.append(accuracy)
        actotal_list = list(zip(*actotal_list))
        if best < np.mean(actotal_list[3]):
            best = np.mean(actotal_list[3])
            best_params = [param1, param2]
    return best_params


def get_likelihood(x, mean, variance):
    if variance == 0:
        return inf
    else:
        return (1 / sqrt(2 * pi * variance)) * exp((-((x - mean)**2)) / (2 * variance))


def make_test(test, train, no_classes):
    classes = group(test, no_classes)
    prob, means, variances = bayes(train, no_classes)
    errors = [1 for i in range(no_classes)]
    total = [0, 0]

    for i in range(len(classes)):
        for x in classes[i]:
            if check_prediction(x, prob, means, variances) != i:
                errors[i] -= 1 / len(classes[i])
                total[0] += 1
            total[1] += 1
    errors.append(1 - (total[0] / total[1]))
    return errors


def check_prediction(x, probs, means, variances):
    temp_list = []
    for prob, mean, variance in zip(probs, means, variances):
        # P(X) * P(X|C)
        temp_list.append(prob * get_likelihood(x[1], mean[0], variance[0]) * get_likelihood(x[2], mean[1], variance[1]))
    maximum = max(temp_list)
    return temp_list.index(maximum)


def bayes(data, no_classes):
    probs = calc_class_probs(data, no_classes)
    classes = group(data, no_classes)
    means, variances = get_means_variances(classes)
    return probs, means, variances


def calc_class_probs(data, no_classes):
    prob = [0 for n in range(0, no_classes)]
    for row in data:
        prob[row[0]] += (1 / len(data))
    return prob


def get_means_variances(data):
    means = []
    variances = []
    for _class in data:
        tab0 = []
        tab1 = []
        for index, arg0, arg1 in _class:
            tab0.append(arg0)
            tab1.append(arg1)
        means.append([np.mean(tab0), np.mean(tab1)])
        variances.append([st.variance(tab0), st.variance(tab1)])
    return means, variances


def draw_graphs(data):
    for param0, param1 in combinations(range(0, len(data[0][1])), 2):
        plt.clf()
        for _class in group(pick_parameters(data, param0, param1), 3):
            index, args0, args1 = list(zip(*_class))
            plt.plot(args0, args1, 'o')
        plt.savefig(f"graphs\\graph{param0}_{param1}.png")


if __name__ == "__main__":
    no_classes = 3  # [1, 2, 3] -> [0, 1, 2]

    data = read_from_file("wine.data")

    # draw_graphs(data)

    train, test = divide(0.8, data)
    param1, param2 = cross_validate(train, 6, no_classes)

    print(f"Best parameters: {param1}, {param2}")
    train = pick_parameters(train, param1, param2)
    test = pick_parameters(test, param1, param2)

    accuracy = make_test(train, test, no_classes)
    print(f"Accuracy 0: {accuracy[0]*100:0.2f}%")
    print(f"Accuracy 1: {accuracy[1]*100:0.2f}%")
    print(f"Accuracy 2: {accuracy[2]*100:0.2f}%")
    print(f"Total accuracy: {accuracy[3]*100:0.2f}%")
