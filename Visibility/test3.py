from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111, projection='3d')


# u = np.linspace(0, 2*np.pi, 100)
# v = np.linspace(0, np.pi, 100)
#
# x = 10*np.outer(np.cos(u), np.sin(v))
# y = 10*np.outer(np.sin(u), np.sin(v))
# z = 10*np.outer(np.ones(np.size(u)), np.cos(v))


# y = np.array([0,0,1,1]) - 0.5
# z = np.array([0,2,2,0])
#
# Y, Z = np.meshgrid(y, z)
# X = np.array(0*Y) - 0.5
# X1 = X + 1
#
# x_top = np.array([0, 0, 1, 1]) - 0.5
# y_top = np.array([0, 1, 1, 0]) - 0.5
# X_top,Y_top = np.meshgrid(x_top, y_top)
# Z_top = np.array(0*Y) + 2
length = 0.5
def plot_cuboid(pos, height, color, ax, alpha):
    y = np.array([0, 0, length, length]) - length/2
    z = np.array([0, height, height, 0])
    Y, Z = np.meshgrid(y, z)
    X = np.array(0 * Y) - length/2
    ax.plot_surface(X + pos[0], Y + pos[1], Z, color=color, alpha=alpha, edgecolors='k')
    ax.plot_surface(Y + pos[0], X + pos[1], Z, color=color, alpha=alpha, edgecolors='k')

    X1 = X + length
    ax.plot_surface(X1 + pos[0], Y + pos[1], Z, color=color, alpha=alpha, edgecolors='k')
    ax.plot_surface(Y + pos[0], X1 + pos[1], Z, color=color, alpha=alpha, edgecolors='k')

    x_top = np.array([0, 0, length, length]) - length/2
    y_top = np.array([0, length, length, 0]) - length/2
    X_top, Y_top = np.meshgrid(x_top, y_top)
    Z_top = np.array(0 * Y) + height
    ax.plot_surface(X_top + pos[0], Y_top + pos[1], Z_top, color=color, alpha=alpha, edgecolors='k')


plot_cuboid([-2, 2], 2, 'b', ax, 1)

plot_cuboid([0, 0], 3, 'r', ax, 1)

a = 1
# ax.plot_surface(X, Y, Z, color='b', alpha=a)
# ax.plot_surface(Y, X, Z, color='b',alpha=a)
# ax.plot_surface(X1, Y, Z, color='b',alpha=a)
# ax.plot_surface(Y,X1,Z, color='b',alpha=a)
# ax.plot_surface(X_top, Y_top,Z_top, color='b',alpha=a)
# ax.plot_surface(Z, X, Y, color='b')

ax.set_xlabel('X')
ax.set_xlim3d(-5, 5)

ax.set_ylabel('Y')
ax.set_ylim3d(-5, 5)

ax.set_zlabel('Z')
ax.set_zlim3d(0, 10)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
plt.show()
