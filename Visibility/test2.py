from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111, projection='3d')
xs = np.arange(0, 10, 0.4)
verts = []
zs = [0.0, 1.0, 2.0, 3.0]
zs = [0]
for z in zs:
    ys = np.random.rand(len(xs))  # creates random ys data points
    ys[0], ys[-1] = 0, 0  # ensure it starts and ends at 0
    verts.append(list(zip(xs, ys)))  # join each xs and ys point into a tuple, then form a list of them and append the
    # list to verts
print(verts)

# xs = [0, 5]
# ys = [5, 5]
# for z in zs:
#     verts.append(list(zip(xs,ys)))

verts = [[(0, 0), (0, 1), (10, 1), (10, 0)]]
verts_3d = [np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([1, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 1]), np.array([1, 0, 1])]
verts_3d = np.array([[np.array([0,0,0]), np.array([0,1,0]), np.array([1,1,0]), np.array([1,0,0]), np.array([0,0,0.5]), np.array([0,1,0.5]), np.array([1,1,0.5]), np.array([1,0,0.5])]])
# poly = PolyCollection(verts)
poly = Poly3DCollection(verts_3d)  #, np.array([1,1,1]), np.array([1,1,2]))
poly.set_3d_properties()
poly.do_3d_projection()
# poly.set_alpha(0.7)
# poly.draw()
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, 10)

ax.set_ylabel('Y')
ax.set_ylim3d(-1, 4)

ax.set_zlabel('Z')
ax.set_zlim3d(0, 1)

plt.show()