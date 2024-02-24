import numpy as np
from Net import Net 
from Layer import Layer, activation_sum
from random import randint

np.random.seed(1)

layer1 = Layer(4, 4)
layer2 = Layer(4, 3)
layer2.activation = activation_sum
layer3 = Layer(3, 0)
layer2.connect(layer1)
layer3.connect(layer2)

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def encode(name):
    if name == 'Iris-setosa':
        return [1, 0, 0]
    elif name == 'Iris-versicolor':
        return [0, 1, 0]
    elif name == 'Iris-virginica':
        return [0, 0, 1]
    else:
        raise Exception("Encoding error")


def get_data():
    data = []
    with open("iris.data", "r") as file:
        for line in file.readlines():
            splitted_line = line.split(",")
            if len(splitted_line) == 5:
                data.append([list(map(float, splitted_line[:4])), encode(splitted_line[4][:-1])])
        return np.array(data)


def predict(x):
    x = layer1(x)
    x = layer2(x)
    return x


def backpropagation(y_pr, y):
    layer3.grad = 2*(y_pr - y)
    layer2.backpropagation()
    layer1.backpropagation()


def create_sets(set1_set2_ratio, data):
    np.random.shuffle(data)
    set1_size = int(set1_set2_ratio*len(data))
    set2_size = len(data) - set1_size
    set1 = data[:set1_size]
    set2 = data[-set2_size:]
    return set1, set2


def classify(array):
    return [1 if element == max(array) else 0 for element in array]


data = get_data()
train_set, test_set = create_sets(0.8, data)
x = train_set[:, 0]
y = train_set[:, 1]

for _ in range(100):
    for i in range(100):
        j = randint(0, 99)
        y_pr = predict(x[j])
        backpropagation(y_pr, y[j])

x_t = test_set[:, 0]
y_t = test_set[:, 1]
error = 0
for i in range(30):
    # print(predict(x[i]), y[i])
    if classify(predict(x[i])) != y[i]:
        error += 1

# print((1 - error/len(test_set))*100)

"""
Using Net
"""
NN = Net([4, 4, 3], True)
data = get_data()
train_set, test_set = create_sets(0.8, data)
x = train_set[:, 0]
y = train_set[:, 1]
NN.train(x, y, 10000, True)

x_t = test_set[:, 0]
y_t = test_set[:, 1]
error = 0
for i in range(30):
    print(NN(x[i]), y[i])
    if classify(NN(x[i])) != y[i]:
        error += 1
print((1 - error/len(test_set))*100)