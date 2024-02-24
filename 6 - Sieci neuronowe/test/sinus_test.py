import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer, activation_sum
from Net import Net
from random import randint
np.random.seed(10)

layer1 = Layer(1, 4)
layer2 = Layer(4, 1)
layer2.activation = activation_sum
layer3 = Layer(1, 0)
layer2.connect(layer1)
layer3.connect(layer2)

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def predict(x):
    x = layer1(x)
    x = layer2(x)
    return x


def backpropagation(y_pr, y):
    layer3.grad = 2*(y_pr - y)
    layer2.backpropagation()
    layer1.backpropagation()


x = np.linspace(-np.pi, np.pi, 100)
x = np.split(x, 100)
y = np.sin(x)

for _ in range(1000):
    for i in range(100):
        j = randint(0, 99)
        y_pr = predict(x[j])
        backpropagation(y_pr, y[j])


pred_y = [predict(argument) for argument in x]
plt.plot(x, y)
plt.plot(x, pred_y)
# plt.show()

"""
Using Net
"""
NN = Net([1, 4, 1], True)
x = np.linspace(-np.pi, np.pi, 100)
x = np.split(x, 100)
y = np.sin(x)
NN.train(x, y, 100000, True)

pred_y = [NN(argument) for argument in x]
plt.plot(x, y)
plt.plot(x, pred_y)
# plt.show()
