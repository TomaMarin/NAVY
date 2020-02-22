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


all_points_change = list(list())


def main_loop(dimension, number_of_points, learning_rate, number_of_epochs, points):
    perceptron = Perceptron(np.random.random((2, 1)), random.uniform(0, 1), dimension, learning_rate)
    for j in range(number_of_epochs):
        predicted_points = list()
        for i in range(number_of_points):
            # if j > number_of_epochs-10:
            predicted_points.append(
                Point(points[i].parameters, calculate_signum(points[i].parameters, perceptron.weights,
                                                             perceptron.bias)), )
            perceptron.weights = calculate_new_weights(perceptron.weights, learning_rate, points[i].value_of_line,
                                                       calculate_signum(points[i].parameters, perceptron.weights,
                                                                        perceptron.bias), points[i].parameters)
            perceptron.bias = perceptron.bias + learning_rate * (
                    points[i].value_of_line - calculate_signum(points[i].parameters, perceptron.weights,
                                                               perceptron.bias))
        all_points_change.append(predicted_points)

    return perceptron


fig, ax = plt.subplots(num=1, figsize=(8, 5))

# fig, ax = plt.figure(num=3, figsize=(8, 5))
u, = plt.plot([], [], '.', color='r', marker="o")
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
plt.title("Iteration: " )
help_x = list()
help_x_d = list()
help_y = list()
help_y_d = list()
all_x_vals = list()
all_y_vals = list()


# u, = plt.scatter(all_x_vals, all_y_vals, marker="o")
# d, = plt.scatter(all_x_vals, all_y_vals, marker="x")


def animate_vals(ite):
    # print(ite, " vals")
    if ite >= len(all_points_change):
        help_x.clear()
        help_y.clear()
        ani_vals.frame_seq = ani_vals.new_frame_seq()
    else:
        help_x.clear()
        help_y.clear()
        for k in all_points_change[999]:
            if k.value_of_line == 1:
                help_x.append(k.parameters[0])
                help_y.append(k.parameters[1])
    u.set_xdata(help_x)
    u.set_ydata(help_y)
    plt.draw()
    return u,


def animate_vals_downwards(ite):
    plt.title("Iteration: " + str(ite))

    if ite >= len(all_points_change):
        help_x_d.clear()
        help_y_d.clear()
        ani_vals.frame_seq = ani_vals.new_frame_seq()
    else:
        help_x_d.clear()
        help_y_d.clear()
        for k in all_points_change[ite]:
            if k.value_of_line < 1:
                help_x_d.append(k.parameters[0])
                help_y_d.append(k.parameters[1])
    d.set_xdata(help_x_d)
    d.set_ydata(help_y_d)
    plt.draw()
    return d,


x_axis = list()
y_axis = list()

dimension = 2
number_of_points = 100
number_of_epochs = 1000
learning_rate = 0.1
points = generate_values_to_learn(dimension, number_of_points)
perct = main_loop(dimension, number_of_points, learning_rate, number_of_epochs, points)
print(perct)



upward_points_x = list()
upward_points_y = list()
downward_points_x = list()
downward_points_y = list()
for i in all_points_change[number_of_epochs - 1]:
    if i.value_of_line == 1:
        upward_points_x.append(i.parameters[0])
        upward_points_y.append(i.parameters[1])
    else:
        downward_points_x.append(i.parameters[0])
        downward_points_y.append(i.parameters[1])

figg, axg = plt.subplots(num=2, figsize=(8, 5))
plt.plot(upward_points_x, upward_points_y, '.', color='r', marker="o")
plt.plot(downward_points_x, downward_points_y, '.', color='g', marker="x")
axg.set(xlim=(-30, 30), ylim=(-40, 40))
plt.plot(x, y1,

         color='red',
         linewidth=1.0,
         linestyle='--'
         )

ani_vals = animation.FuncAnimation(fig, animate_vals, interval=200, repeat=False)
animate_vals_downwards = animation.FuncAnimation(fig, animate_vals_downwards, interval=200, repeat=False)

plt.show()
