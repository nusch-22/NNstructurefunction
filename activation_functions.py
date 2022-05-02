import matplotlib.pyplot as plt
import numpy as np
from numpy import tanh


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear(x):
    return x


def relu(x):
    return np.maximum(0, x)


def plot_activations():
    x = np.linspace(-2, 2, 100)
    plt.plot(x, sigmoid(x), label="Sigmoid")
    plt.plot(x, relu(x), label="ReLU")
    plt.plot(x, tanh(x), label="Tanh")
    plt.plot(x, linear(x), label="Linear", linestyle="dashed")
    plt.legend()
    plt.grid()
    plt.savefig("activations.png")


func_str = "exp"

x = 5

y = eval("np." + func_str + "(5)")
print(y)
