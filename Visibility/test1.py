from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Cuboid:
    def __init__(self, pos, size, side_length):
        self.pos = pos
        self.size = size
        self.len = side_length


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111, projection='3d')
x = [1, 1, 1]
ax.scatter(1, 1, 1)
# plt.grid()
plt.show()
