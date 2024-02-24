import numpy as np


def MSE_error(list1, list2):
    return np.square(list1 - list2)


def arctan(array):
    return np.arctan(array) * 2 / np.pi


def arctan_grad(array):
    return 1 / (1 + np.square(array))


def activation_sum(array):
    return array


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def sigmoid_grad(array):
    return sigmoid(array) * (1 - sigmoid(array))


class Layer:
    def __init__(self, input_size, output_size, beta):
        self.weights = np.random.uniform(-1 / np.sqrt(input_size), 1 / np.sqrt(input_size), input_size * output_size)
        self.weights = np.reshape(self.weights, (input_size, output_size))
        self.biases = np.random.uniform(-1 / np.sqrt(input_size), 1 / np.sqrt(input_size), output_size)
        self.values = None
        self.grad = None  # to nie gradient, ale wartość, na której opiera się wyliczanie go, a grad się ładniej pisze
        self.activation = sigmoid
        self.activation_grad = sigmoid_grad
        self.next_layer = None
        self.previous_layer = None
        self.beta = beta

    def __call__(self, input):
        """
        Calculates values of neurones and returns them
        """
        self.values = input
        return self.activation(self.sum())

    def sum(self):
        return np.matmul(self.values, self.weights) + self.biases

    def backpropagation(self):
        if self.previous_layer is not None:
            self.grad = np.matmul(self.next_layer.grad, self.weights.T) * self.previous_layer.activation_grad(self.previous_layer.sum())
        self.weights -= self.beta * np.outer(self.values, self.next_layer.grad)
        self.biases -= self.beta * self.next_layer.grad

    def connect(self, prev_layer):
        self.previous_layer = prev_layer
        prev_layer.next_layer = self
