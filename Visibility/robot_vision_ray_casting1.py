from skgeom import *
import skgeom as sg
from matplotlib import pyplot as plt
import numpy as np
from torch import tensor


class Cylinder:
    def __init__(self, pos, size, radius):
        self.pos = pos
        self.size = size
        self.radius = radius
        self.visibility = "not visible"
        self.segments = []

    def approx_circle(self, edges):
        points = []
        theta = 2 * np.pi / edges
        for i in range(edges):
            x = self.pos[0] + self.radius * np.cos(i * theta)
            y = self.pos[1] + self.radius * np.sin(i * theta)
            pos = sg.Point2(x, y)
            points.append(pos)

        segments = []
        for idx, value in enumerate(points):
            s = sg.Segment2(points[idx], points[(idx + 1) % len(points)])
            segments.append(s)
        self.segments = segments


object_positions = {'Large positions': [tensor([2, 2])], 'Medium positions': [tensor([1, 1])],
                    'Small positions': [tensor([1, 2])]}

objects = []
for position in object_positions['Large positions']:
    objects.append(Cylinder(pos=position.numpy(), size='L', radius=0.3))

for position in object_positions['Medium positions']:
    objects.append(Cylinder(pos=position.numpy(), size='M', radius=0.3))

for position in object_positions['Small positions']:
    objects.append(Cylinder(pos=position.numpy(), size='S', radius=0.3))

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
def approx_circle(centre, radius, edges):
    points = []
    theta = 2 * np.pi / edges
    for i in range(edges):
        x = centre[0] + radius * np.cos(i * theta)
        y = centre[1] + radius * np.sin(i * theta)
        pos = sg.Point2(x, y)
        points.append(pos)

    segments = []
    for idx, value in enumerate(points):
        s = sg.Segment2(points[idx], points[(idx + 1) % len(points)])
        segments.append(s)
    return segments


for circle in objects:
    circle.approx_circle(36)

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

camera_range = 10
fov = np.pi/8  # radians

#

print('ehladk')
