import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors

# representation of given objects by rgb colors
TREE_COLOR = (51, 153, 51)
FIRE_COLOR = (255, 153, 51)
GROUND_COLOR = (0, 0, 0)
cmap = colors.ListedColormap(np.array([TREE_COLOR, FIRE_COLOR, GROUND_COLOR]))

# array needed to find neighbours of point in 2d area
neighbourhood = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# initialization of area with given probability to create forest
def init_area(init_tree_generation_probability, size):
    area = list()
    for i in range(size):
        area_row = list()
        for j in range(size):
            if np.random.uniform() < init_tree_generation_probability:
                area_row.append(TREE_COLOR)
            else:
                area_row.append(GROUND_COLOR)
        area.append(area_row)
    return area


# function to find whether given tree has any neighbour on fire
def is_neighbour_tree_on_fire(x, y, array):
    for dx, dy in neighbourhood:
        if array[y + dy][x + dx] == FIRE_COLOR:
            return True
    return False


# simulation function
def simulation(p, f, area):
    current_area = copy.deepcopy(area)
    # iterating through whole area
    for ii in range(0, len(area) - 1):
        for ij in range(0, len(area) - 1):
            # first rule, if there is fire, it becomes ground(empty)
            if current_area[ii][ij] == FIRE_COLOR:
                current_area[ii][ij] = GROUND_COLOR
            # second rule, if point is a ground(empty), then for give prob. p, new tree is created
            elif current_area[ii][ij] == GROUND_COLOR and np.random.uniform(0, 1) < p:
                current_area[ii][ij] = TREE_COLOR
            # third rule, if point is a tree, then for give prob. f,  tree catches on fire
            elif current_area[ii][ij] == TREE_COLOR and np.random.uniform(0.0, 1.0) < f:
                current_area[ii][ij] = FIRE_COLOR
            # fourth rule, if point is a tree, and has any neighbour on fire, it catches on fire also
            elif current_area[ii][ij] == TREE_COLOR and is_neighbour_tree_on_fire(ii, ij, current_area):
                current_area[ii][ij] = FIRE_COLOR
    return current_area


# probs for spawning trees and fires
p, f = 0.005, 0.0005
area = init_area(0.75, 100)

fig = plt.figure(figsize=(25 / 3, 6.25))
ax = fig.add_subplot(111)
ax.set_axis_off()
im = ax.imshow(area, cmap=cmap, )  # , interpolation='nearest')


# The animation function
def animate(i):
    im.set_data(animate.X)
    animate.X = simulation(p, f, animate.X)


# Bind  grid to the identifier X in the animation's  namespace.
animate.X = area

interval = 40
anim = animation.FuncAnimation(fig, animate, interval=interval)
plt.show()
