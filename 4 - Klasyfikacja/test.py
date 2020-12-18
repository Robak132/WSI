import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


def get_data():
    data = []
    with open("iris.data", "r") as file:
        for line in file.readlines():
            splitted_line = line.split(",")
            if len(splitted_line) == 5:
                data.append([list(map(float, splitted_line[:4])), encode(splitted_line[4][:-1])])
        return data


def encode(name):
    if name == 'Iris-setosa':
        return [1, -1, -1]
    elif name == 'Iris-versicolor':
        return [-1, 1, -1]
    elif name == 'Iris-virginica':
        return [-1, -1, 1]
    else:
        raise Exception("Encoding error")


def create_train_test_sets(train_test_ratio, data):
    np.random.shuffle(data)
    train_set_size = int(train_test_ratio * len(data))
    test_set_size = len(data) - train_set_size
    train_set = data[:train_set_size]
    test_set = data[-test_set_size:]
    return train_set, test_set


def split_data(data):
    class1 = []
    class2 = []
    class3 = []
    for row in data:
        if row[1][0] == 1:
            class1.append(row)
        elif row[1][1] == 1:
            class2.append(row)
        elif row[1][2] == 1:
            class3.append(row)
        else:
            raise Exception("https://tenor.com/view/thanos-impossible-marvel-shocked-gif-15104180")
    return class1, class2, class3


def make_2D_set(feature1, feature2, index, data):
    data_set = np.array([[row[0][feature1], row[0][feature2], row[1][index]] for row in data])
    return data_set


def make_4D_set(index, data):
    data_set = np.array([[row[0][0], row[0][1], row[0][2], row[0][3], row[1][index]] for row in data])
    return data_set


def linear_SVM(train_set):
    global TRAIN_SET
    global RECORD_SIZE
    global LAMBDA
    LAMBDA = 0.5
    TRAIN_SET = train_set
    a = TRAIN_SET[:, :4]
    constraint = NonlinearConstraint(constr, 0, np.inf)
    res = minimize(target, np.random.random(5), constraints=constraint)
    w = res.x[:4]
    b = res.x[4]
    return w, b


def target(w):
    return LAMBDA * np.linalg.norm(w[:4], 2) + ksi(w).sum()


def ksi(w):
    return np.maximum(0, 1 - ((np.matmul(TRAIN_SET[:, :4], w[:4]) - w[4]) * TRAIN_SET[:, 4]))


def f(x, w, b):
    return np.matmul(np.transpose(w), x) - b


def constr(w):
    return ((np.matmul(TRAIN_SET[:, :4], w[:4]) - w[4]) * TRAIN_SET[:, 4]) + ksi(w) - 1


def threshold_unipolar_function(x, a):
    if (x >= a):
        return 1
    elif (x < a):
        return -1


def predict(x, w_12, w_23, w_31, b_12, b_23, b_31):
    x_lpredict = [threshold_unipolar_function(np.matmul(w_12, x) - b_12, 0),
                  threshold_unipolar_function(np.matmul(w_23, x) - b_23, 0),
                  threshold_unipolar_function(np.matmul(w_31, x) - b_31, 0)]
    if (x_lpredict[2] == -1) and (x_lpredict[0] == 1):
        return [1, -1, -1]
    if (x_lpredict[0] == -1) and (x_lpredict[1] == 1):
        return [-1, 1, -1]
    if (x_lpredict[1] == -1) and (x_lpredict[2] == 1):
        return [-1, -1, 1]


if __name__ == "__main__":
    data = get_data()
    train_set, test_set = create_train_test_sets(0.9, data)
    set1, set2, set3 = split_data(train_set)
    w_12, b_12 = linear_SVM(make_4D_set(0, set1 + set2))
    print(w_12)
    print(b_12)
    """
    w_23, b_23 = linear_SVM(make_4D_set(1, set2 + set3))
    w_31, b_31 = linear_SVM(make_4D_set(2, set3 + set1))

    tset1, tset2, tset3 = split_data(test_set)
    errors1 = 0
    errors2 = 0
    errors3 = 0

    for row in tset1:
        if (predict(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [1, -1, -1]):
            errors1 += 1
    print(f"Class Iris-setosa: accuracy {(1 - errors1/len(tset1))*100:.2f}%")

    for row in tset2:
        if (predict(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [-1, 1, -1]):
            errors2 += 1
    print(f"Class Iris-versicolor: accuracy {(1 - errors2/len(tset2))*100:.2f}%")

    for row in tset3:
        if (predict(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [-1, -1, 1]):
            errors3 += 1
    print(f"Class Iris-virginica: accuracy {(1 - errors3/len(tset3))*100:.2f}%")
    print(f"Total accuracy {(1 - (errors1 + errors2 + errors3)/(len(tset1) + len(tset2) + len(tset3)))*100:.2f}%")
    """