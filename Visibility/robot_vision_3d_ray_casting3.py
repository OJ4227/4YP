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
        self.segments_obstructing = []

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

    def check_visibility(self, visible_segments):
        for s in self.segments:
            if s in visible_segments or s.opposite() in visible_segments:
                self.visibility = 'visible'
                self.segments_obstructing.append(s)


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


class Ray:
    def __init__(self, camera, dir):
        """

        :param camera: camera object, which will be where the ray is cast from
        :param dir: np array of the 3D direction vector
        """
        self.pos = camera.pos
        self.dir = dir

    def draw_ray(self, ax):
        x = [self.pos[0], self.pos[0] + self.dir[0]]
        y = [self.pos[1], self.pos[1] + self.dir[1]]
        z = [0, 0 + self.dir[2]]
        ax.plot(x, y, z)

# Load in objects
object_positions = {'Large positions': [tensor([2, 2])], 'Medium positions': [tensor([1, 1])],
                    'Small positions': [tensor([1, 2])]}

# Initialise cuboids
objects = []
for position in object_positions['Large positions']:
    objects.append(Cuboid(pos=position.numpy(), size='L', side_length=0.3))

for position in object_positions['Medium positions']:
    objects.append(Cuboid(pos=position.numpy(), size='M', side_length=0.3))

for position in object_positions['Small positions']:
    objects.append(Cuboid(pos=position.numpy(), size='S', side_length=0.3))

# Create first figure
fig = plt.figure(1, figsize=(6, 6))

#### Do 2D visibility calculations

# Initialise the camera
camera = Camera(np.array([1, -1]), optic_range=5, fov=np.pi / 3)
camera.compute_viewing_region(num_sides=15)

# Compute the 2d segments for each cuboid
for cuboid in objects:
    cuboid.compute_2d_segments()

# Create the arrangement
arr = arrangement.Arrangement()

# Add the camera's viewing boundaries
for s in camera.viewing_region_segments:
    arr.insert(s)

# Add the cuboids' boundaries
for circle in objects:
    for s in circle.segments:
        arr.insert(s)

# Draw the arrangement
for he in arr.halfedges:  # each edge is separated into two half edges
    draw.draw(he.curve(), marker='')  # curve() turns the half-edges into segments?

# Initialise the rotational sweep
vs = RotationalSweepVisibility(arr)
q = Point2(camera.pos[0], camera.pos[1])

# Compute the visibility surface
face = arr.find(q)
vx = vs.compute_visibility(q, face)

# Create a list containing all the visible segments
visible_segments = []
for v in vx.halfedges:
    visible_segments.append(v.curve())
    draw.draw(v.curve(), color='red', visible_point=False)

# Draw the end points of the segments that are obstructing the camera's view
for point in vx.vertices:
    draw.draw(point.point(), marker='x')

# Check whether each object is visible and which edges are the ones obstructing the camera's view
for cuboid in objects:
    cuboid.check_visibility(visible_segments)


draw.draw(q, color='magenta')

print('ehladk')
####
# Initialise second figure
fig2 = plt.figure(2, figsize=(6, 6))
ax2 = fig2.add_subplot(111, projection='3d')

# Plot each cuboid
for cuboid in objects:
    cuboid.plot_cuboid(ax2, alpha=1)

# Plot the camera
camera.plot_camera(ax2)


# Initialise and plot the ray
direction = np.array([0, 1, 1])
ray = Ray(camera, direction)
ray.draw_ray(ax2)

# Specify ax2 labels, dimensions and ticks
ax2.set_xlabel('X')
ax2.set_xlim3d(-1, 4)

ax2.set_ylabel('Y')
ax2.set_ylim3d(-1, 4)

ax2.set_zlabel('Z')
ax2.set_zlim3d(0, 10)

ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.yaxis.set_major_locator(MultipleLocator(1))
plt.show()



