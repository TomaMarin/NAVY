import matplotlib
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
    if val > 0.0:
        return 1
    elif val == 0.00:
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


all_points_change = list(list())

predicted_test_points = list()


def learning_phase(dimension, number_of_points, learning_rate, number_of_epochs, points):
    perceptron = Perceptron(np.random.random((2, 1)), random.uniform(0, 1), dimension, learning_rate)
    for j in range(number_of_epochs):
        for i in range(number_of_points):
            perceptron.weights = calculate_new_weights(perceptron.weights, learning_rate, points[i].value_of_line,
                                                       calculate_signum(points[i].parameters, perceptron.weights,
                                                                        perceptron.bias), points[i].parameters)
            perceptron.bias = perceptron.bias + learning_rate * (
                    points[i].value_of_line - calculate_signum(points[i].parameters, perceptron.weights,
                                                               perceptron.bias))

    return perceptron


def testing_phase(perceptron: Perceptron, number_of_points, test_points):
    for k in range(number_of_points):
        predicted_test_points.append(
            Point(test_points[k].parameters, calculate_signum(test_points[k].parameters, perceptron.weights,
                                                              perceptron.bias)))
    perceptron.weights = calculate_new_weights(perceptron.weights, learning_rate, test_points[k].value_of_line,
                                               calculate_signum(test_points[k].parameters, perceptron.weights,
                                                                perceptron.bias), test_points[k].parameters)
    perceptron.bias = perceptron.bias + learning_rate * (
            test_points[k].value_of_line - calculate_signum(test_points[k].parameters, perceptron.weights,
                                                            perceptron.bias))


fig, ax = plt.subplots(num=1, figsize=(8, 5))

# fig, ax = plt.figure(num=3, figsize=(8, 5))
u = plt.plot([], [], '.', color='r', marker="o")
d, = plt.plot([], [], '.', color='g', marker="x")
# plt.figure(num=3, figsize=(8, 5))
ax.set(xlim=(-30, 30), ylim=(-40, 40))
x = np.linspace(-25, 20)
y1 = function(x)
plt.plot(x, y1,
         color='red',
         linewidth=1.0,
         linestyle='--'
         )
# ax.plot()
plt.title("Prediction: ")

x_axis = list()
y_axis = list()

dimension = 2
number_of_points = 100
number_of_epochs = 2000
learning_rate = 0.15
test_points = generate_values_to_learn(2, 100)
points = generate_values_to_learn(dimension, number_of_points)
perct = learning_phase(dimension, number_of_points, learning_rate, number_of_epochs, points)
testing_phase(perct, number_of_points, test_points)
print(perct)

upward_points_x = list()
upward_points_y = list()
downward_points_x = list()
downward_points_y = list()

middle_points_x = list()
middle_points_y = list()
for i in test_points:
    if i.value_of_line == 1:
        upward_points_x.append(i.parameters[0])
        upward_points_y.append(i.parameters[1])
    elif i.value_of_line == 0:
        middle_points_x.append(i.parameters[0])
        middle_points_y.append(i.parameters[1])
    else:
        downward_points_x.append(i.parameters[0])
        downward_points_y.append(i.parameters[1])

figg, axg = plt.subplots(num=2, figsize=(8, 5))
plt.plot(upward_points_x, upward_points_y, '.', color='r', marker="o")
plt.plot(middle_points_x, middle_points_y, '.', color='b', marker="o")
plt.plot(downward_points_x, downward_points_y, '.', color='g', marker="x")
axg.set(xlim=(-30, 30), ylim=(-40, 40))
plt.plot(x, y1,

         color='red',
         linewidth=1.0,
         linestyle='--'
         )

upward_points_x_predicted = list()
upward_points_y_predicted = list()

downward_points_x_predicted = list()
downward_points_y_predicted = list()

middle_points_predicted_x = list()
middle_points_predicted_y = list()
for i in predicted_test_points:
    if i.value_of_line == 1:
        upward_points_x_predicted.append(i.parameters[0])
        upward_points_y_predicted.append(i.parameters[1])
    elif i.value_of_line == 0:
        middle_points_predicted_x.append(i.parameters[0])
        middle_points_predicted_y.append(i.parameters[1])
    else:
        downward_points_x_predicted.append(i.parameters[0])
        downward_points_y_predicted.append(i.parameters[1])

fig, axg = plt.subplots(num=1, figsize=(8, 5))
plt.plot(upward_points_x_predicted, upward_points_y_predicted, '.', color='r', marker="o")
plt.plot(middle_points_predicted_x, middle_points_predicted_y, '.', color='b', marker="o")
plt.plot(downward_points_x_predicted, downward_points_y_predicted, '.', color='g', marker="x")
axg.set(xlim=(-30, 30), ylim=(-40, 40))
plt.plot(x, y1,

         color='red',
         linewidth=1.0,
         linestyle='--'
         )

for i in range(len(test_points)):
    if test_points[i].value_of_line == predicted_test_points[i].value_of_line:
        print(i, " success value: ", test_points[i].value_of_line)
    else:
        print(i, "failure, expected: ", test_points[i].value_of_line, " predicted: ",
              predicted_test_points[i].value_of_line)
        print(i, "values: ", test_points[i])

plt.show()
