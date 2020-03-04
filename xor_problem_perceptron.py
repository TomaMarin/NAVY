import numpy as np
import random
import matplotlib.pyplot as plt

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])


def sigmoid_derivative(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main_loop(number_of_hidden_layers, number_of_output_layers, number_of_epochs, inputs, expected_outputs,
              learning_rate):
    hidden_layers_weights = np.random.random(size=(2, number_of_hidden_layers))
    hidden_layers_biases = np.random.random(size=(1, number_of_hidden_layers))
    output_layers_weights = np.random.random(size=(number_of_hidden_layers, number_of_output_layers))
    output_layers_biases = np.random.random(size=(1, number_of_output_layers))

    print("hidden_layers_biases begin: ", hidden_layers_biases)
    print("output_layers_biases begin: ", output_layers_biases)

    for i in range(number_of_epochs):
        hidden_layer_net = np.matmul(inputs, hidden_layers_weights) + hidden_layers_biases
        hidden_layers_out = sigmoid(hidden_layer_net)

        output_layers_net = np.matmul(hidden_layers_out, output_layers_weights) + output_layers_biases
        prediction_out = sigmoid(output_layers_net)

        error_array = expected_outputs - prediction_out
        prediction_out_delta = error_array * sigmoid_derivative(prediction_out)

        error_array_hidden_layer = np.matmul(prediction_out_delta, output_layers_weights.T)
        hidden_layers_delta = error_array_hidden_layer * sigmoid_derivative(hidden_layers_out)

        hidden_layers_weights += np.matmul(inputs.T, hidden_layers_delta) * learning_rate
        # hidden_layers_biases += np.sum(hidden_layers_delta, axis=0, keepdims=True)
        hidden_layers_biases = hidden_layers_biases + np.sum(learning_rate * hidden_layers_delta, axis=0, keepdims=True)

        output_layers_weights += np.matmul(hidden_layers_out.T, prediction_out_delta) * learning_rate
        # output_layers_biases += np.sum(prediction_out_delta, axis=0, keepdims=True)
        output_layers_biases = output_layers_biases + np.sum(learning_rate * prediction_out_delta, axis=0,
                                                             keepdims=True)

    print("hidden_layers_biases end: ", hidden_layers_biases)
    print("output_layers_biases end: ", output_layers_biases)
    print("Predicted output: ", *prediction_out)
    return


epochs = 5000
learning_rate = 0.2
number_of_hidden_layers = 2
number_of_output_layers = 1
main_loop(number_of_hidden_layers, number_of_output_layers, epochs, inputs, expected_output, learning_rate)
