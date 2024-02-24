from Layer import Layer, activation_sum
from random import randint


class Net:
    def __init__(self, array, linear, beta):
        self.beta = beta
        self.last_linear = linear
        self.create_layers(array)

    def create_layers(self, array):
        self.layers = []
        for i in range(len(array)):
            if i == len(array) - 1:  # check if i is the last element in the array
                self.layers.append(Layer(array[i], 0, self.beta))
            else:
                self.layers.append(Layer(array[i], array[i + 1], self.beta))
            if (i != 0):
                self.layers[-1].connect(self.layers[-2])
        if self.last_linear:
            self.layers[-2].activation = activation_sum  # arctan is default function

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

    def backpropagation(self, y_pr, y):
        if self.last_linear:
            self.layers[-1].grad = 2 * (y_pr - y)  # it would be more elegant to assign activation grad to last layer
        else:
            self.layers[-1].grad = 2 * (y_pr - y) * (self.layers[-1].activation_grad(self.layers[-2].sum()))
        for layer in reversed(self.layers[:-1]):
            layer.backpropagation()

    def train(self, x, y, iterations, random):
        for _ in range(iterations):
            if random:
                # taking random element and learning on it
                j = randint(0, len(x) - 1)
                y_pr = self(x[j])
                self.backpropagation(y_pr, y[j])
            else:
                # iterating on every element - not best
                for j in range(len(x)):
                    y_pr = self(x[j])
                    self.backpropagation(y_pr, y[j])
