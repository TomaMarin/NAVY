import numpy as np
import random
import matplotlib.pyplot as plt

inputs = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])


class Layer:
    net = 0
    out = 0
    bias = 0
    error = 0

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calc_net(self, out_array):
        for i in range(len(self.weights)):
            self.net += out_array[i] * self.weights[i]
        self.net += self.bias * 1

    def calc_out(self):
        self.out = np.exp(self.net) / (np.exp(self.net) + 1)

    def __repr__(self):
        return "Layer with weights " + str(self.weights) + " ,net " + str(self.net) + " ,out " + str(
            self.out) + ", bias " + str(self.bias) + "  and error " + str(self.error)


def sigmoid_derivative(x):
    return x * (1 - x)


def sigmoid(expected, predicted):
    result = 1 / 2 * np.power((expected[2] - predicted), 2)
    return result


def calculate_new_weights(weights, learning_rate, partial_derivative):
    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * partial_derivative
    return weights


def weight_recalculation(weight, learning_constant, total_error):
    new_weight = 0.0
    new_weight = weight * learning_constant


def main_loop(number_of_epochs, inputs, learning_rate):
    hidden_layers = list()
    hidden_layers.append(Layer(np.random.random((2, 1)), random.uniform(0, 1)))
    hidden_layers.append(Layer(np.random.random((2, 1)), random.uniform(0, 1)))

    output_layer = (Layer(np.random.random((2, 1)), random.uniform(0, 1)))

    total_error = 0
    for i in range(number_of_epochs):
        for j in range(len(inputs)):
            for k in hidden_layers:
                k.calc_net(inputs[j])
                k.calc_out()

            output_layer.calc_net([hidden_layers[0].out, hidden_layers[1].out])
            output_layer.calc_out()
            print("predicted output: ", output_layer.out)
            error = inputs[j][2] - output_layer.out
            output_layer.error = sigmoid(inputs[j], output_layer.out)
            d_predicted_output = error * sigmoid(inputs[j], output_layer.out)
            error_hidden_layer = d_predicted_output.dot(output_layer.weights.T)
            for l in range(len(output_layer.weights)):
                for m in range(len(hidden_layers)):
                    output_layer.weights[l] += hidden_layers[m].weights[l] * learning_rate
        print("\n")


learning_rate = 0.15
main_loop(20, inputs, learning_rate)
