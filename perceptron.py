import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm, animation


def activation_function(parameters):
    return parameters[1] - (4 * parameters[0] - 5)


def function(x):
    return 4 * x - 5


class Perceptron:
    def __init__(self, weights, bias, dimension, learning_rate):
        self.weights = weights
        self.bias = bias
        self.dimension = dimension
        self.learning_rate = learning_rate

    def __repr__(self):
        return "Perceptron with weights on " + str(self.weights) + " and bias val " + str(self.bias)


class Point:
    def __init__(self, parameters, value_of_line):
        self.parameters = parameters
        self.value_of_line = value_of_line


def __repr__(self):
    return "Point with parameters " + str(self.parameters) + " and value_of_line " + str(self.value_of_line)


def set_value_of_line(parameters):
    if activation_function(parameters) > 0.0:
        return 1
    elif activation_function(parameters) == 0.0:
        return 0
    else:
        return -1


def signum_function(val):
    if val > 0:
        return 1
    elif val == 0:
        return 0
    else:
        return -1


def generate_values_to_learn(dimension, number_of_points):
    points = list()
    for i in range(number_of_points):
        point_params = random.sample(range(-25, 25), dimension)
        new_point = Point(point_params, set_value_of_line(point_params))
        points.append(new_point)
    return points


def sum_function(parameters, weights, bias):
    total_val = bias
    for i in range(len(parameters)):
        total_val += parameters[i] * weights[i]
    return total_val


def calculate_signum(parameters, weights, bias):
    return signum_function(sum_function(parameters, weights, bias))


def calculate_new_weights(weights, learning_rate, expected, predicted, parameters):
    for i in range(len(weights)):
        weights[i] = weights[i] + learning_rate * (expected - predicted) * parameters[i]
    return weights


predicted_points = list()


def main_loop(dimension, number_of_points, learning_rate, number_of_epochs, points):
    perceptron = Perceptron(np.random.random((2, 1)), random.uniform(0, 1), dimension, learning_rate)
    for j in range(number_of_epochs):
        for i in range(number_of_points):
            if j == number_of_epochs - 1:
                predicted_points.append(
                    Point(points[i].parameters, calculate_signum(points[i].parameters, perceptron.weights,
                                                                 perceptron.bias)), )
                print("expected ", points[i].value_of_line, "predicted ",
                      calculate_signum(points[i].parameters, perceptron.weights,
                                       perceptron.bias))
            perceptron.weights = calculate_new_weights(perceptron.weights, learning_rate, points[i].value_of_line,
                                                       calculate_signum(points[i].parameters, perceptron.weights,
                                                                        perceptron.bias), points[i].parameters)
            perceptron.bias = perceptron.bias + learning_rate * (
                    points[i].value_of_line - calculate_signum(points[i].parameters, perceptron.weights,
                                                               perceptron.bias))

    return perceptron


x_axis = list()
y_axis = list()

dimension = 2
number_of_points = 100
number_of_epochs = 1000
learning_rate = 0.1
points = generate_values_to_learn(dimension, number_of_points)
perct = main_loop(dimension, number_of_points, learning_rate, number_of_epochs, points)
print(perct)

plt.figure(num=3, figsize=(8, 5))

x = np.linspace(-5, 10, 75)
y1 = function(x)

upward_points_x = list()
upward_points_y = list()
downward_points_x = list()
downward_points_y = list()
for i in predicted_points:
    if i.value_of_line == 1:
        upward_points_x.append(i.parameters[0])
        upward_points_y.append(i.parameters[1])
    else:
        downward_points_x.append(i.parameters[0])
        downward_points_y.append(i.parameters[1])

plt.plot(x, y1,
         color='red',
         linewidth=1.0,
         linestyle='--'
         )
plt.scatter(upward_points_x, upward_points_y, marker="o")
plt.scatter(downward_points_x, downward_points_y, marker="x")
plt.show()
