from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8  # The gravitational acceleration (m.s-2).
L1 = 1.0  # length of pendulum 1
L2 = 1.0  # length of pendulum 2
M1 = 1.0  # mass of pendulum 1
M2 = 1.0  # mass of pendulum 2

# init angles and angle velocities
theta1 = np.rad2deg(2 * np.pi / 6)
theta2 = np.rad2deg(5 * np.pi / 8)
theta1_velocity = 0.0
theta2_velocity = 0.0


# function to calculate double dot theta 1 a 2(acceleration), returns tuple object of velocities and accelerations
def get_derivative(state, t):
    theta1, theta1_velocity, theta2, theta2_velocity = state
    # double dot theta 1 formula
    pendulum1_acceleration = (M2 * G * np.sin(theta2) * np.cos(theta1 - theta2)
                              - M2 * np.sin(theta1 - theta2) * (L1 * theta1_velocity ** 2 * np.cos(
                theta1 - theta2) + L2 * theta2_velocity ** 2)
                              - (M1 + M2) * G * np.sin(theta1)) \
                             / (L1 * (M1 + M2 * np.sin(theta1 - theta2) ** 2))

    # double dot theta 2 formula
    pendulum2_acceleration = ((M1 + M2) * (L1 * theta1_velocity ** 2 * np.sin(theta1 - theta2)
                                           - G * np.sin(theta2) + G * np.sin(theta1) * np.cos(theta1 - theta2))
                              + M2 * L2 * theta2_velocity ** 2 * np.sin(theta1 - theta2) * np.cos(theta1 - theta2)) \
                             / (L2 * (M1 + M2 * np.sin(theta1 - theta2) ** 2))
    return theta1_velocity, pendulum1_acceleration, theta2_velocity, pendulum2_acceleration


# create a time array sampled by 0.05
dt = 0.05
t = np.arange(0, 20, dt)

# initial state
state_0 = [theta1, theta1_velocity, theta2, theta2_velocity]

od = integrate.odeint(get_derivative, state_0, t)

# calculating coordinates for both pendulums
x1 = L1 * sin(od[:, 0])
y1 = -L1 * cos(od[:, 0])

x2 = L2 * sin(od[:, 2]) + x1
y2 = -L2 * cos(od[:, 2]) + y1

# visualisation
fig = plt.figure()
ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 1), )
plt.axis('off')

line, = ax.plot([], [], 'o-', lw=2, c="black")
points = ax.scatter([], [], s=8, c="r")
p_x = []
p_y = []


# init function of animation
def init():
    line.set_data([], [])
    p_x = []
    p_y = []
    points.set_offsets(np.c_[p_x, p_y])
    return line, points


# animation function
def animate(i):
    this_x = [0, x1[i], x2[i]]
    this_y = [0, y1[i], y2[i]]
    p_x.append(x2[i])
    p_y.append(y2[i])
    # deletes old movement track
    if len(p_x) > 100:
        p_x.pop(0)
        p_y.pop(0)
    points.set_offsets(np.c_[p_x, p_y])
    line.set_data(this_x, this_y)
    return line, points


speed = 40
# ani = animation.FuncAnimation(fig, animate, range(1, len(y)), interval=dt * speed, blit=True, init_func=init)
ani = animation.FuncAnimation(fig, animate, range(1, len(od)), interval=speed, blit=True, init_func=init)
plt.show()
