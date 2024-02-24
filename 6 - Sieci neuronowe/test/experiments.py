import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer, activation_sum
from random import randint
np.random.seed(10)


lista = [0, 1, 2, 3, 4, 5, 6]
print(lista[-1])
for number in reversed(lista[:-1]):
    print(number)

# layer1 = Layer(1, 4)
# layer2 = Layer(4, 1)
# layer2.activation = activation_sum
# layer3 = Layer(1, 0)
# layer2.connect(layer1)
# layer3.connect(layer2)

# np.set_printoptions(precision=5)
# np.set_printoptions(suppress=True)


# def predict(x):
#     x = layer1(x)
#     x = layer2(x)
#     return x


# def backpropagation(y_pr, y):
#     layer3.grad = 2*(y_pr - y)
#     layer2.backpropagation()
#     layer1.backpropagation()

# x = np.linspace(-np.pi, np.pi, 100)
# x = np.split(x, 100)
# print(x)
# y = np.sin(x)

# for _ in range(10000):
#     for i in range(1):
#         j = randint(0, 99)
#         y_pr = predict(x[j])
#         backpropagation(y_pr, y[j])



# pred_y = [predict(argument) for argument in x]
# plt.plot(x, y)
# plt.plot(x, pred_y)
# plt.show()



"""
Proste testy na feedforward
"""
# x = np.ones(3)
# test1 = np.array([0.77132064, 0.49850701, 0.16911084])
# print(np.matmul(x, layer1.weights) + layer1.biases)
# print(np.arctan(layer1.sum())*2/np.pi)


"""
Przykładowa konstrukcja i działanie sieci
"""

# layer1 = Layer(3, 3)
# layer2 = Layer(3, 5)
# layer3 = Layer(5, 1)
# layer1.next_layer = layer2
# layer2.previous_layer = layer1
# layer3 = Layer(5, 1)
# layer2.next_layer = layer3
# layer3.previous_layer = layer2
# layer3.activation = activation_sum
# layer4 = Layer(1,0) #zbiera wyniki
# layer3.next_layer = layer4


# arguments = np.linspace(-np.pi, np.pi, 2000)
# train_arguments = np.linspace(np.pi/2, np.pi*3/4, 500)
# y = np.sin(arguments)
# for i in range(100000):
#     input = [np.random.choice(arguments)]
#     x = layer1(input)
#     x = layer2(x)
#     x = layer3(x)
#     layer4.values = x

#     r_output = np.sin(input)
#     gradient = 2*(x - r_output)
#     layer4.grad = gradient
#     layer3.backpropagation()
#     layer2.backpropagation()
#     layer1.backpropagation()
#     if (i%10000 == 0):
#         print(i)
        # check(i)


# x_ax = []
# y_ax = []

# train = [[0,[0,0,1]], [1,[0,1,1]], [1,[1,0,1]], [0,[1,1,1]]]
# for i in range(10000):
#     input = train[np.random.randint(0,4)]
#     x = layer1(input[1])
#     x = layer2(x)
#     x = layer3(x)
#     layer4.values = x

#     r_output = np.array([input[0]])
#     gradient = 2*(x - r_output)
#     layer4.grad = gradient
#     layer3.backpropagation()
#     layer2.backpropagation()
#     layer1.backpropagation()
#     # print(MSE_error(r_output, x))
#     x_ax.append(i)
#     y_ax.append(MSE_error(r_output, x))


# plt.plot(x_ax, y_ax)
# plt.legend()
# plt.show()