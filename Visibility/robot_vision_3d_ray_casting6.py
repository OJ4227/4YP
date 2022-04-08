from skgeom import *
from torch import tensor
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator

# Objects are cuboids, plots the cuboids in 3d grid, created method to calculate ray intersection with any of the planes
# of the cuboids

# side 1 is the side facing along the x axis with the lowest y value (the left side), they then get numerated as you
# move around the cuboid clockwise

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
        self.segments_2d = []
        self.obstructing_segments = []
        self.face_planes = {}

    def compute_2d_segments(self):
        x = self.len / 2
        y = self.len / 2

        segments = [
            Segment2(Point2(self.pos[0] - x, self.pos[1] - y), Point2(self.pos[0] - x, self.pos[1] + y)),
            Segment2(Point2(self.pos[0] - x, self.pos[1] + y), Point2(self.pos[0] + x, self.pos[1] + y)),
            Segment2(Point2(self.pos[0] + x, self.pos[1] + y), Point2(self.pos[0] + x, self.pos[1] - y)),
            Segment2(Point2(self.pos[0] + x, self.pos[1] - y), Point2(self.pos[0] - x, self.pos[1] - y)),
        ]
        self.segments_2d = segments

    def plot_cuboid(self, ax, alpha):
        y = np.array([0, 0, self.len, self.len]) - self.len / 2
        z = np.array([0, self.height, self.height, 0])
        Y, Z = np.meshgrid(y, z)
        X = np.array(0 * Y) - self.len / 2
        ax.plot_surface(X + self.pos[0], Y + self.pos[1], Z, color=self.color, alpha=alpha, edgecolors='k')
        ax.plot_surface(Y + self.pos[0], X + self.pos[1], Z, color=self.color, alpha=alpha, edgecolors='k')

        X1 = X + self.len
        ax.plot_surface(X1 + self.pos[0], Y + self.pos[1], Z, color=self.color, alpha=alpha, edgecolors='k')
        ax.plot_surface(Y + self.pos[0], X1 + self.pos[1], Z, color=self.color, alpha=alpha, edgecolors='k')

        x_top = np.array([0, 0, self.len, self.len]) - self.len / 2
        y_top = np.array([0, self.len, self.len, 0]) - self.len / 2
        X_top, Y_top = np.meshgrid(x_top, y_top)
        Z_top = np.array(0 * Y) + self.height
        ax.plot_surface(X_top + self.pos[0], Y_top + self.pos[1], Z_top, color=self.color, alpha=alpha, edgecolors='k')

    def check_visibility(self, visible_segments):
        for s in self.segments_2d:
            if s in visible_segments or s.opposite() in visible_segments:
                self.visibility = 'visible'
                self.obstructing_segments.append(s)

    def compute_plane_equations(self):
        p1_top = np.array(
            [self.segments_2d[0].source().x(), self.segments_2d[0].source().y(), self.height]).astype(float)
        p2_top = np.array(
            [self.segments_2d[0].target().x(), self.segments_2d[0].target().y(), self.height]).astype(float)

        normal_top = np.array([0, 0, 1])
        face = f'top'

        xlims_top = [p1_top[0], p1_top[0] + self.len]
        ylims_top = [min(p1_top[1], p2_top[1]), min(p1_top[1], p2_top[1]) + self.len]
        zlims_top = [self.height, self.height]
        self.face_planes[face] = {'point': p1_top,
                                  'normal': normal_top,
                                  'xlims': xlims_top,
                                  'ylims': ylims_top,
                                  'zlims': zlims_top}

        for idx, segment in enumerate(self.segments_2d):
            p1 = np.array([segment.source().x(), segment.source().y(), 0]).astype(float)
            p2 = np.array([segment.target().x(), segment.target().y(), 0]).astype(float)
            p3 = np.array([segment.source().x(), segment.source().y(), self.height]).astype(float)

            normal = np.cross(p3 - p1, p2 - p1)
            xlims = np.array([min(p1[0], p2[0]), max(p1[0], p2[0])])
            ylims = np.array([min(p1[1], p2[1]), max(p1[1], p2[1])])
            zlims = np.array([0, self.height])

            face = f'side {idx+1}'
            self.face_planes[face] = {'point': p1,
                                      'normal': normal,
                                      'xlims': xlims,
                                      'ylims': ylims,
                                      'zlims': zlims,
                                      'segment': segment}


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
        ax.scatter(self.pos[0], self.pos[1], self.pos[2])


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
        z = [self.pos[2], self.pos[2] + self.dir[2]]
        ax.plot(x, y, z)


# Load in objects
object_positions = {'Large positions': [tensor([2, 2])], 'Medium positions': [tensor([1, 2])],
                    'Small positions': [tensor([1, 1])]}

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
camera = Camera(np.array([1, -1, 0]), optic_range=5, fov=np.pi / 3)
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
for cuboid in objects:
    for s in cuboid.segments_2d:
        arr.insert(s)

# Draw the arrangement
for he in arr.halfedges:  # each edge is separated into two half edges
    draw.draw(he.curve(), color="#1f77b4", marker='')  # curve() turns the half-edges into segments?

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

# Along each obstructing segment of each object, construct a certain number of rays in that direction
list_of_rays = []
for cuboid in objects:
    for segment in cuboid.obstructing_segments:

        point1 = np.array([segment.source().x(), segment.source().y(), cuboid.height + 0.01])
        point2 = np.array([segment.target().x(), segment.target().y(), cuboid.height + 0.01])
        l1 = np.linspace(0, 1, 3)

        ray_points = point1 + (point2 - point1) * l1[:, None]

        for point in ray_points:
            direction = point - camera.pos
            direction = direction.astype(float)
            normalized_direction = direction / np.linalg.norm(direction)
            # Need to normalise this so it's a unit direction vector, watch the video
            # but change the intersection stuff to intersecting a plane. May not actually need to do that could see if
            # it intersects 2D segment and verify that the segment could exist at the height it would need to be to be
            # visible to the ray (would need to use trig to calculate the necessary height)
            list_of_rays.append(Ray(camera, normalized_direction))

        print('heeeey sugar')


# Compute the plane equations for each face of each cuboid to be used in the ray casting
for cuboid in objects:
    cuboid.compute_plane_equations()

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def ray_cast(ray, face_planes):
    ndotu = face_planes['normal'].dot(ray.dir)
    if abs(ndotu) < (1e-6):
        return None
    else:
        w = ray.pos - face_planes['point']
        si = -face_planes['normal'].dot(w) / ndotu
        Psi = w + si * ray.dir + face_planes['point']
        return Psi


def ray_cast1(ray, plane_dict):
    ndotu = plane_dict['normal'].dot(ray.dir)
    if abs(ndotu) < (1e-6):
        return None
    else:
        w = ray.pos - plane_dict['point']
        si = -plane_dict['normal'].dot(w) / ndotu
        intersect = w + si * ray.dir + plane_dict['point']

        return si, intersect


def ray_cast2(ray, cuboid):
    for plane_dict in cuboid.face_planes:
        ndotu = plane_dict['normal'].dot(ray.dir)
        if abs(ndotu) < (1e-6):
            continue
        else:
            w = ray.pos - plane_dict['point']
            si = -plane_dict['normal'].dot(w) / ndotu
            intersect = w + si * ray.dir + plane_dict['point']

            return si, intersect

# t1 = ray_cast(list_of_rays[-2], objects[1].face_planes['side 4'])
# ray = Ray(camera, t1-camera.pos)
t2, intersect = ray_cast1(list_of_rays[-2], objects[1].face_planes['side 4'])
a = camera.pos + list_of_rays[-2].dir * t2
if np.array_equiv(a, intersect):
    print('True')
ray = Ray(camera, camera.pos + list_of_rays[-2].dir * t2 - camera.pos)



# Plot the rays
# for ray in list_of_rays:
#     ray.draw_ray(ax2)
list_of_rays[-2].draw_ray(ax2)
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
