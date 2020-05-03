import numpy as np
import matplotlib.pyplot as plt


# function for given formula x_{n+1} = a*x_n*(1-x_n)
def logistic_map_function(r, x):
    return r * x * (1 - x)


# function to create bifurcation diagram
# n - amount of points
# iterations - number of iterations for loop
# r - points of biotic potential (positive constants)
def bifurcation_diagram(n, iterations, last, r):
    # initialize x points
    x = 1e-5 * np.ones(n)
    for i in range(iterations):
        # calculate x points
        x = logistic_map_function(r, x)
        # visualize only last 100 iterations
        if i >= (iterations - last):
            ax1.plot(r, x, ',k', alpha=.25)



n = 10000
r = np.linspace(2.5, 4.0, n)
iterations = 1000
last = 100
fig, ax1 = plt.subplots()

bifurcation_diagram(n, iterations, last, r)

ax1.set_xlim(2.5, 4)
# r points are on x axis and x points are on y axis
plt.show()
