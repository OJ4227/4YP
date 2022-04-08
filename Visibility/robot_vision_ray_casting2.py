from skgeom import *
import skgeom as sg
from matplotlib import pyplot as plt
import numpy as np
from torch import tensor

# Objects are cuboids


class Cuboid:
    def __init__(self, pos, size, side_length):
        self.pos = pos
        self.size = size
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


class Camera:
    def __init__(self, pos, optic_range, fov):
        self.pos = pos
        self.range = optic_range
        self.fov = fov


object_positions = {'Large positions': [tensor([2, 2])], 'Medium positions': [tensor([1, 1])],
                    'Small positions': [tensor([1, 2])]}

objects = []
for position in object_positions['Large positions']:
    objects.append(Cuboid(pos=position.numpy(), size='L', side_length=0.3))

for position in object_positions['Medium positions']:
    objects.append(Cuboid(pos=position.numpy(), size='M', side_length=0.3))

for position in object_positions['Small positions']:
    objects.append(Cuboid(pos=position.numpy(), size='S', side_length=0.3))

M = 4
outer = [
    Segment2(Point2(-M, -M), Point2(-M, M)),
    Segment2(Point2(-M, M), Point2(M, M)),
    Segment2(Point2(M, M), Point2(M, -M)),
    Segment2(Point2(M, -M), Point2(-M, -M)),
]


# segments = [
#     Segment2(Point2(0, 0), Point2(0, 4)),
#     Segment2(Point2(2, 4), Point2(8, 4)),
#     Segment2(Point2(3, 4), Point2(-8, 0)),
#     Segment2(Point2(1, 0), Point2(0, 5)),
# ]


##

for cuboid in objects:
    cuboid.compute_2d_segments()

##
arr = arrangement.Arrangement()

for s in outer:
    arr.insert(s)

# for s in segments:
#     arr.insert(s)
for circle in objects:
    for s in circle.segments:
        arr.insert(s)

for he in arr.halfedges:  # each edge is separated into two half edges
    draw.draw(he.curve(), marker='')  # curve() turns the half-edges into segments?

# plt.show()

###
vs = RotationalSweepVisibility(arr)
q = Point2(1, -1)
camera = Camera(np.array([1, -1]), optic_range=5, fov=np.pi/3)

camera_points = []
camera_points.append(Point2(camera.pos[0], camera.pos[1]-0.1))
num_sides = 10
thetas = np.linspace(-camera.fov/2, camera.fov/2, num_sides + 1)
for theta in thetas:
    x = camera.pos[0] + camera.range*np.sin(theta)
    y = camera.pos[1] + camera.range*np.cos(theta)
    camera_points.append(Point2(x, y))

face = arr.find(q)
vx = vs.compute_visibility(q, face)

# for he in arr.halfedges:
#     draw.draw(he.curve(), visible_point=False)

visible_segments = []

for v in vx.halfedges:
    visible_segments.append(v.curve())
    draw.draw(v.curve(), color='red', visible_point=False)

for point in vx.vertices:
    draw.draw(point.point(), marker='x')

for circle in objects:
    for s in circle.segments:
        if s in visible_segments or s.opposite() in visible_segments:
            circle.visibility = 'visible'
            break

draw.draw(q, color='magenta')

plt.show()


#

print('ehladk')
