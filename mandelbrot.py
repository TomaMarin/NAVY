import numpy as np
import matplotlib.pyplot as plt

iterations = 500
m = 2


# function implementation by definition  z_0  = 0, z_{n+1} = z_n^2 + c
def mandelbrot(c, m, iterations):
    z = 0
    n = 0
    # loop with condition that for set exists number that satisfies condtion |z_n| <= m and max iterations
    while abs(z) <= m and n < iterations:
        z = z ** 2 + c
        n += 1
    return n


# create X, Y  points for ranges (-2,0.5) and (-1,1) by 0.007 point scale
# by knowing that set is inside of circle with center at 0 and radius of 2, and its approx. extents
X = np.arange(-2, 0.5, 0.007)
Y = np.arange(-1, 1, 0.007)
Z = np.zeros((len(Y), len(X)))

# iterating through sets of X and Y values
for i in range(len(Y)):
    # this is just to show a progress of iterations, whole algorithm takes a little more time to complete
    progress = round((i / len(Y)) * 100)
    if progress % 25 == 0:
        print("Progress:" + str(progress))
    for j in range(len(X)):
        # calculation of c complex value by multiplying imaginary value with Y value and adding X value
        c = X[j] + 1j * Y[i]
        # calculating Z value of graph of  mandelbrot for given c
        Z[i, j] = mandelbrot(c, m, iterations)

# visualisation
plt.imshow(Z, cmap='Greys')
plt.axis("off")
plt.show()
