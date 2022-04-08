from skgeom import *
from torch import tensor
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator


# Objects are cuboids, plots the cuboids in 3d grid
large = 3
medium = 2
small = 1


class Cuboid:
    def __init__(self, pos, size, side_length):
        self.pos = pos
        self.size = size
        if size == 'L':
            self.height = large
            self.color = 'r'
        elif size == 'M':
            self.height = medium
            self.color = 'm'
        else:
            self.height = small
            self.color = 'b'
        self.len = side_length
        self.visibility = "not visible"
        self.segments = []

    def compute_2d_segments(self):
        x = self.len/2
        y = self.len/2

        segments = [
            Segment2(Point2(self.pos[0]-x, self.pos[1]-y), Point2(self.pos[0]-x, self.pos[1]+y)),
            Segment2(Point2(self.pos[0]-x, self.pos[1]+y), Point2(self.pos[0]+x, self.pos[1]+y)),
            Segment2(Point2(self.pos[0]+x, self.pos[1]+y), Point2(self.pos[0]+x, self.pos[1]-y)),
            Segment2(Point2(self.pos[0]+x, self.pos[1]-y), Point2(self.pos[0]-x, self.pos[1]-y)),
        ]
        self.segments = segments

    def plot_cuboid(self, ax, alpha):
        y = np.array([0, 0, self.len, self.len]) - self.len/2
        z = np.array([0, self.height, self.height, 0])
        Y, Z = np.meshgrid(y, z)
        X = np.array(0 * Y) - self.len/2
        ax.plot_surface(X + self.pos[0], Y + self.pos[1], Z, color=self.color, alpha=alpha, edgecolors='k')
        ax.plot_surface(Y + self.pos[0], X + self.pos[1], Z, color=self.color, alpha=alpha, edgecolors='k')

        X1 = X + self.len
        ax.plot_surface(X1 + self.pos[0], Y + self.pos[1], Z, color=self.color, alpha=alpha, edgecolors='k')
        ax.plot_surface(Y + self.pos[0], X1 + self.pos[1], Z, color=self.color, alpha=alpha, edgecolors='k')

        x_top = np.array([0, 0, self.len, self.len]) - self.len/2
        y_top = np.array([0, self.len, self.len, 0]) - self.len/2
        X_top, Y_top = np.meshgrid(x_top, y_top)
        Z_top = np.array(0 * Y) + self.height
        ax.plot_surface(X_top + self.pos[0], Y_top + self.pos[1], Z_top, color=self.color, alpha=alpha, edgecolors='k')


class Camera:
    def __init__(self, pos, optic_range, fov):
        self.pos = pos
        self.range = optic_range
        self.fov = fov
        self.viewing_region_segments = []

    def compute_viewing_region(self, num_sides):
        num_sides = 15

        camera_points = [Point2(self.pos[0], self.pos[1] - 0.001)]  # first point is the camera location, cannot
        # place it exactly on the camera as this throws up an error

        thetas = np.linspace(-self.fov / 2, self.fov / 2, num_sides + 1)
        for theta in thetas:
            x = self.pos[0] + self.range * np.sin(theta)
            y = self.pos[1] + self.range * np.cos(theta)
            camera_points.append(Point2(x, y))

        segments = []
        for idx, value in enumerate(camera_points):
            s = Segment2(camera_points[idx], camera_points[(idx + 1) % len(camera_points)])
            segments.append(s)

        self.viewing_region_segments = segments


    def plot_camera(self, ax):
        ax.scatter(self.pos[0], self.pos[1], 0)



object_positions = {'Large positions': [tensor([2, 2])], 'Medium positions': [tensor([1, 1])],
                    'Small positions': [tensor([1, 2])]}

objects = []
for position in object_positions['Large positions']:
    objects.append(Cuboid(pos=position.numpy(), size='L', side_length=0.3))

for position in object_positions['Medium positions']:
    objects.append(Cuboid(pos=position.numpy(), size='M', side_length=0.3))

for position in object_positions['Small positions']:
    objects.append(Cuboid(pos=position.numpy(), size='S', side_length=0.3))


fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111, projection='3d')

for cuboid in objects:
    cuboid.plot_cuboid(ax, alpha=1)

camera = Camera(np.array([1, -1]), optic_range=5, fov=np.pi/3)
camera.plot_camera(ax)
ax.set_xlabel('X')
ax.set_xlim3d(-1, 4)

ax.set_ylabel('Y')
ax.set_ylim3d(-1, 4)

ax.set_zlabel('Z')
ax.set_zlim3d(0, 10)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
plt.show()