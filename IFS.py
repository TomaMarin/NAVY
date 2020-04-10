import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# method for affine transformation  by given formula, everything needs to be reshaped to be calculated correctly
#  |x'|   | a b c |   |x|   |j|
# w|y'| = | d e f | * |y| + |k|
#  |z'|   | g h i |   |z|   |l|
def transform_model(transformation, jkl_vector, position):
    trans_arr = (transformation).reshape(3, 3)
    new_transformation = np.dot(trans_arr, np.array(position).reshape(3, 1)) + np.array(jkl_vector).reshape(3, 1)
    return new_transformation


# method to get  index of random transformation by given probability, works correctly for p=0.25
def get_transformation_index_by_random(p):
    random_number = random.random()
    if random_number < p:
        return 0
    elif random_number < 2 * p:
        return 1
    elif random_number < 3 * p:
        return 2
    else:
        return 3


# lists to save positions of transformation
positions_x = list()
positions_y = list()
positions_z = list()


# main loop of algorithm, firstly it generates random positions
# then in loop it gets random index for model transformation and its jkl values
# after that new positions are given by method transform_model
# finally positions are appended to lists
def main_loop(model, jkl_vector_array, iterations, p):
    position_xyz = np.random.random_sample((3,))

    for _ in range(iterations):
        random_number = get_transformation_index_by_random(p)
        transformation = model[random_number]
        position_xyz = transform_model(transformation, jkl_vector_array[random_number], position_xyz)
        positions_x.append(position_xyz[0])
        positions_y.append(position_xyz[1])
        positions_z.append(position_xyz[2])


# models  and its jkl vectors
model1 = np.array([np.array([0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05]),
                   np.array([0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24]),
                   np.array([-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24]),
                   np.array([0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84])])
jkl_vector_array1 = np.array([[0.00, 0.00, 0.00], [0.00, 0.80, 0.00], [0.00, 0.22, 0.00], [0.00, 0.80, 0.00]])

model2 = np.array([np.array([0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05]),
                   np.array([0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45]),
                   np.array([-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45]),
                   np.array([0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49])])

jkl_vector_array2 = np.array([[0.00, 0.00, 0.00], [0.00, 1.00, 0.00], [0.00, 1.25, 0.00], [0.00, 2.00, 0.00]])

fig = plt.figure()
ax = fig.gca(projection='3d')

probability = 0.25
iterations = 35000
# un/comment all 3 commands to get transformation  one of the models

main_loop(model1, jkl_vector_array1, iterations, probability)
ax.view_init(40, 260)
s = 3

# main_loop(model2, jkl_vector_array2, iterations, probability)
# ax.view_init(30, 30)
# s = 0.5

# visualization takes a little time and even manipulating the plot is little bit tricky because of many points
ax.scatter(positions_x, positions_y, positions_z, s=s, zdir='z')
plt.show()
