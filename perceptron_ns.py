import matplotlib
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm, animation
import PerceptronTask


def activation_function(parameters):
    return parameters[1] - (4 * parameters[0] - 5)


def function_for_decision_boundary_Line(y, perceptron, w1, w2):
    return -(perceptron.bias - w1 * y) / w2


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return s, ds


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


def signum_function(val):
    if val > 0.0:
        return 1
    else:
        return 0


# def generate_values_to_learn(dimension, number_of_points):
#     points = list()
#     for i in range(number_of_points):
#         point_params = random.sample(range(-25, 25), dimension)
#         new_point = Point(point_params, set_value_of_line(point_params))
#         points.append(new_point)
#     return points


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


def training_phase(dimension, number_of_points, learning_rate, number_of_epochs, points):
    perceptron = Perceptron(np.random.random((2, 1)), random.uniform(0, 1), dimension, learning_rate)
    for j in range(number_of_epochs):
        for i in range(number_of_points):
            perceptron.weights = calculate_new_weights(perceptron.weights, learning_rate, points[i].output,
                                                       calculate_signum(points[i].inputs, perceptron.weights,
                                                                        perceptron.bias), points[i].inputs)
            print(points[i].output - calculate_signum(points[i].inputs, perceptron.weights,
                                                      perceptron.bias))
            perceptron.bias = perceptron.bias + learning_rate * (
                    points[i].output - calculate_signum(points[i].inputs, perceptron.weights,
                                                        perceptron.bias))

    return perceptron


def testing_phase(perceptron: Perceptron, number_of_points, test_points):
    for k in range(number_of_points):
        output = calculate_signum(test_points[k].inputs, perceptron.weights, perceptron.bias)
        test_points[k].output = output
    # perceptron.weights = calculate_new_weights(perceptron.weights, learning_rate, test_points[k].output,
    #                                            calculate_signum(test_points[k].inputs, perceptron.weights,
    #                                                             perceptron.bias), test_points[k].inputs)
    # perceptron.bias = perceptron.bias + learning_rate * (
    #         test_points[k].output - calculate_signum(test_points[k].inputs, perceptron.weights,
    #                                              perceptron.bias))


x_axis = list()
y_axis = list()
perc_task = PerceptronTask.PerceptronTask()
perc_task.parse_xml('obdelnik_rozsah.xml', 'perceptronTask')

train_points = perc_task.trainElements
test_points = perc_task.testElements
dimension = 2
number_of_points = len(train_points)
number_of_epochs = 2000
learning_rate = 0.3

perct = training_phase(dimension, number_of_points, learning_rate, number_of_epochs, train_points)

testing_phase(perct, len(test_points), test_points)
print(perct)

k = - perct.weights[0] / perct.weights[1]
c = - perct.bias / perct.weights[1]
point_1x = k * 0.0 + c
point_2x = k * 1.0 + c
xy_min = -5
xy_max = 15

point_1y = 1.0 / k * 0.0 - c / k
point_2y = 1.0 / k * 1.0 - c / k
x = [0.0, 1.0, point_1y, point_2y, xy_max, xy_min, 1.0 / k * xy_max - c / k, 1.0 / k * xy_min - c / k]
y1 = [point_1x, point_2x, 0.0, 1.0, k * xy_max + c, k * xy_min + c, xy_max, xy_min]
fig, ax = plt.subplots(num=1, figsize=(8, 5))

upward_points_x = list()
upward_points_y = list()
downward_points_x = list()
downward_points_y = list()

for i in train_points:
    if i.output == 0:
        upward_points_x.append(i.inputs[0])
        upward_points_y.append(i.inputs[1])

    else:
        downward_points_x.append(i.inputs[0])
        downward_points_y.append(i.inputs[1])

upward_points_x_predicted = list()
upward_points_y_predicted = list()

downward_points_x_predicted = list()
downward_points_y_predicted = list()

for i in test_points:
    if i.output == 0:
        upward_points_x_predicted.append(i.inputs[0])
        upward_points_y_predicted.append(i.inputs[1])
    else:
        downward_points_x_predicted.append(i.inputs[0])
        downward_points_y_predicted.append(i.inputs[1])

fig, axg = plt.subplots(num=1, figsize=(8, 5))
plt.plot(upward_points_x_predicted, upward_points_y_predicted, '.', color='r', marker="|")
plt.plot(downward_points_x_predicted, downward_points_y_predicted, '.', color='r', marker="_")
plt.plot(upward_points_x, upward_points_y, '.', color='g', marker="|")
plt.plot(downward_points_x, downward_points_y, '.', color='g', marker="_")
axg.set(xlim=(-5, 15), ylim=(-5, 15))
# axg.set(xlim=(perc_task.min_max_vals_desc[0].min, perc_task.min_max_vals_desc[0].max), ylim=(perc_task.min_max_vals_desc[1].min, perc_task.min_max_vals_desc[1].min))
plt.plot(x, y1,

         color='orange',
         linewidth=1.0,
         linestyle='--'
         )

for i in range(len(test_points)):
    # if test_points[i].output == predicted_test_points[i].output:
    # print(i, " success value: ", test_points[i].output)
    # else:
    #     print(i, "failure, expected: ", test_points[i].output, " predicted: ",
    #           predicted_test_points[i].output)
    print(i, "values: ", test_points[i].inputs, " predicted output: ", test_points[i].output)

plt.show()
