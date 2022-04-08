from skgeom import *
import skgeom as sg
from matplotlib import pyplot as plt
import numpy as np
from torch import tensor


# Objects are cuboids, introduced camera class. could add the functionality to the camera of vertical fov and
# orientation, atm the vertical fov is 180 degrees and the orientation is always forwards


class Cuboid:
    def __init__(self, pos, size, side_length):
        self.pos = pos
        self.size = size
        self.len = side_length
        self.visibility = "not visible"
        self.segments = []
        self.segments_obstructing = []

    def compute_2d_segments(self):
        x = self.len / 2
        y = self.len / 2

        segments = [
            Segment2(Point2(self.pos[0] - x, self.pos[1] - y), Point2(self.pos[0] - x, self.pos[1] + y)),
            Segment2(Point2(self.pos[0] - x, self.pos[1] + y), Point2(self.pos[0] + x, self.pos[1] + y)),
            Segment2(Point2(self.pos[0] + x, self.pos[1] + y), Point2(self.pos[0] + x, self.pos[1] - y)),
            Segment2(Point2(self.pos[0] + x, self.pos[1] - y), Point2(self.pos[0] - x, self.pos[1] - y)),
        ]
        self.segments = segments

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
            s = sg.Segment2(camera_points[idx], camera_points[(idx + 1) % len(camera_points)])
            segments.append(s)

        self.viewing_region_segments = segments


object_positions = {'Large positions': [tensor([2, 2])], 'Medium positions': [tensor([1, 1])],
                    'Small positions': [tensor([1, 2])]}

objects = []
for position in object_positions['Large positions']:
    objects.append(Cuboid(pos=position.numpy(), size='L', side_length=0.3))

for position in object_positions['Medium positions']:
    objects.append(Cuboid(pos=position.numpy(), size='M', side_length=0.3))

for position in object_positions['Small positions']:
    objects.append(Cuboid(pos=position.numpy(), size='S', side_length=0.3))

# M = 4
# outer = [
#     Segment2(Point2(-M, -M), Point2(-M, M)),
#     Segment2(Point2(-M, M), Point2(M, M)),
#     Segment2(Point2(M, M), Point2(M, -M)),
#     Segment2(Point2(M, -M), Point2(-M, -M)),
# ]

camera = Camera(np.array([1, -1]), optic_range=5, fov=np.pi / 3)

##
camera.compute_viewing_region(15)

for cuboid in objects:
    cuboid.compute_2d_segments()

##
arr = arrangement.Arrangement()

# for s in outer:
#     arr.insert(s)

for s in camera.viewing_region_segments:
    arr.insert(s)

for circle in objects:
    for s in circle.segments:
        arr.insert(s)

for he in arr.halfedges:  # each edge is separated into two half edges
    draw.draw(he.curve(), marker='')  # curve() turns the half-edges into segments?

# plt.show()

###
vs = RotationalSweepVisibility(arr)
q = Point2(camera.pos[0], camera.pos[1])

face = arr.find(q)
vx = vs.compute_visibility(q, face)

visible_segments = []

for v in vx.halfedges:
    visible_segments.append(v.curve())
    draw.draw(v.curve(), color='red', visible_point=False)

    v.source().point()
    v.target().point()

for point in vx.vertices:
    draw.draw(point.point(), marker='x')

for cuboid in objects:
    cuboid.check_visibility(visible_segments)
    # for s in cuboid.segments:
    #     if s in visible_segments or s.opposite() in visible_segments:
    #         circle.visibility = 'visible'
    #         cuboid.segments_obstructing.append(s)

draw.draw(q, color='magenta')

plt.show()

print('ehladk')
