import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def initialize_parameters(layer_dimensions):
    parameters = {}
    for i in range(1, len(layer_dimensions)):
        parameters[f"W{i}"] = np.random.randn(layer_dimensions[i], layer_dimensions[i - 1]) * 0.01
        parameters[f"b{i}"] = np.zeros((layer_dimensions[i], 1))
    return parameters


def activate(x, activation_function='sigmoid'):
    if activation_function == 'sigmoid':
        return sigmoid(x)
    elif activation_function == 'tanh':
        return tanh(x)
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")


def activate_derivative(x, activation_function='sigmoid'):
    if activation_function == 'sigmoid':
        return sigmoid_derivative(x)
    elif activation_function == 'tanh':
        return tanh_derivative(x)
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")


def forward_propagation(X, parameters, activation_function='sigmoid'):
    cache = {"A0": X}
    for i in range(1, len(parameters) // 2 + 1):
        W = parameters[f"W{i}"]
        b = parameters[f"b{i}"]
        A_prev = cache[f"A{i - 1}"]
        Z = np.dot(W, A_prev) + b
        A = activate(Z, activation_function)
        cache[f"Z{i}"] = Z
        cache[f"A{i}"] = A
    return cache[f"A{len(parameters) // 2}"], cache


def backward_propagation(AL, Y, cache, parameters, activation_function='sigmoid'):
    grads = {}
    m = Y.shape[1]

    dAL = 2 * (AL - Y)
    for i in range(len(parameters) // 2, 0, -1):
        W = parameters[f"W{i}"]
        A = cache[f"A{i}"]
        Z = cache[f"Z{i}"]

        dZ = dAL * activate_derivative(A, activation_function)
        dW = (1 / m) * np.dot(dZ, A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dAL = np.dot(W.T, dZ)

        grads[f"dW{i}"] = dW
        grads[f"db{i}"] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    for i in range(1, len(parameters) // 2 + 1):
        parameters[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
        parameters[f"b{i}"] -= learning_rate * grads[f"db{i}"]
    return parameters

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1/m) * np.sum((AL - Y) ** 2)
    return cost

def train_neural_network(X, Y, layer_dimensions, learning_rate, activation_function='sigmoid', num_iterations=1000):
    parameters = initialize_parameters(layer_dimensions)

    for _ in range(num_iterations):
        AL, cache = forward_propagation(X, parameters, activation_function)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, cache, parameters, activation_function)
        parameters = update_parameters(parameters, grads, learning_rate)

        if _ % 100 == 0:
            print(f"Iteration {_}, Cost: {cost}")

    return parameters

# Example usage:
# Assuming X and Y are your input and output data, and layer_dimensions is a list specifying the number of neurons in each layer.
# X, Y, layer_dimensions = ...
# trained_parameters = train_neural_network(X, Y, layer_dimensions, learning_rate=0.01, activation_function='tanh')