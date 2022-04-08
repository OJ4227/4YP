from skgeom import *
from torch import tensor
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import generate_object_positions


# Now outputs the state variables and the robot's perception of the state, cleaned up some of the code

# Camera cannot be higher up than the smallest object

# Objects are cuboids, plots the cuboids in 3d grid, implemented the full 3D ray casting for a general problem

# side 1 is the side facing along the x axis with the lowest y value (the left side), they then get numerated as you
# move around the cuboid clockwise


class Cuboid:
    def __init__(self, pos, size, side_length):
        self.pos = pos
        self.coords = np.array([pos[1], pos[0]])
        self.size = size
        if size == 'L':
            self.height = large
            self.color = 'c'
        elif size == 'M':
            self.height = medium
            self.color = 'm'
        elif size == 'S':
            self.height = small
            self.color = 'b'
        else:  # This is for empty positions
            self.height = None
            self.color = None
        self.len = side_length
        self.visibility = "not visible"
        self.segments_2d = []
        self.obstructing_segments = []
        self.face_planes = {}

    def compute_2d_segments(self):
        x = self.len / 2
        y = self.len / 2

        segments = [  # Swapped these position indices around
            Segment2(Point2(self.coords[0] - x, self.coords[1] - y), Point2(self.coords[0] - x, self.coords[1] + y)),
            Segment2(Point2(self.coords[0] - x, self.coords[1] + y), Point2(self.coords[0] + x, self.coords[1] + y)),
            Segment2(Point2(self.coords[0] + x, self.coords[1] + y), Point2(self.coords[0] + x, self.coords[1] - y)),
            Segment2(Point2(self.coords[0] + x, self.coords[1] - y), Point2(self.coords[0] - x, self.coords[1] - y)),
        ]
        self.segments_2d = segments

    def plot_cuboid(self, ax, alpha):
        y = np.array([0, 0, self.len, self.len]) - self.len / 2
        z = np.array([0, self.height, self.height, 0])
        Y, Z = np.meshgrid(y, z)
        X = np.array(0 * Y) - self.len / 2
        ax.plot_surface(X + self.coords[0], Y + self.coords[1], Z, color=self.color, alpha=alpha, edgecolors='k')
        ax.plot_surface(Y + self.coords[0], X + self.coords[1], Z, color=self.color, alpha=alpha, edgecolors='k')

        X1 = X + self.len
        ax.plot_surface(X1 + self.coords[0], Y + self.coords[1], Z, color=self.color, alpha=alpha, edgecolors='k')
        ax.plot_surface(Y + self.coords[0], X1 + self.coords[1], Z, color=self.color, alpha=alpha, edgecolors='k')

        x_top = np.array([0, 0, self.len, self.len]) - self.len / 2
        y_top = np.array([0, self.len, self.len, 0]) - self.len / 2
        X_top, Y_top = np.meshgrid(x_top, y_top)
        Z_top = np.array(0 * Y) + self.height
        ax.plot_surface(X_top + self.coords[0], Y_top + self.coords[1], Z_top, color=self.color, alpha=alpha,
                        edgecolors='k')

    def check_visibility(self, visible_segments):
        for s in visible_segments:
            if s in self.segments_2d or s.opposite() in self.segments_2d:  # check if the entire segment visible and if
                # so append to the obstructing segments
                self.visibility = 'visible'
                self.obstructing_segments.append(s)
            else:  # check if both the source and the target of the visible segment lies on one of the cuboid's 2d
                # segments then append the 2d segment to the obstructing_segments list. This is for the case where the
                # camera cannot see the entire segment due to its viewing angle being too small
                for t in self.segments_2d:
                    if t.has_on(s.source()) and t.has_on(s.target()):
                        self.visibility = 'visible'
                        self.obstructing_segments.append(t)

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

            face = f'side {idx + 1}'
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
    def __init__(self, camera, dir, anchor):
        """

        :param camera: camera object, which will be where the ray is cast from
        :param dir: np array of the 3D direction vector
        """
        self.pos = camera.pos
        self.dir = dir
        self.anchor = anchor

    def draw_ray(self, ax, multiplier=1):
        x = [self.pos[0], self.pos[0] + multiplier * self.dir[0]]
        y = [self.pos[1], self.pos[1] + multiplier * self.dir[1]]
        z = [self.pos[2], self.pos[2] + multiplier * self.dir[2]]
        ax.plot(x, y, z)

    def compute_face_intersect(self, plane_dict, camera):
        ndotu = plane_dict['normal'].dot(self.dir)
        if abs(ndotu) < 1e-6:
            return 1e7, None  # Multiplier output cannot be None because we use it later in a less than statement
        else:
            w = self.pos - plane_dict['point']
            multiplier = -plane_dict['normal'].dot(w) / ndotu
            intersect = w + multiplier * self.dir + plane_dict['point']

            # Include 1e-6 in all the limits so that they are not overly strict and don't accept intersections that they should
            if (plane_dict['xlims'][0] - 1e-6 <= intersect[0] <= plane_dict['xlims'][1] + 1e-6) and (
                    plane_dict['ylims'][0] - 1e-6 <= intersect[1] <= plane_dict['ylims'][1] + 1e-6) and (
                    plane_dict['zlims'][0] - 1e-6 <= intersect[2] <= plane_dict['zlims'][1] + 1e-6) and (
                    np.linalg.norm(
                        intersect - camera.pos) <= camera.range):  # Checks the ray isn't too long for the camera to see
                return multiplier, intersect
            else:
                return 1e7, None  # Multiplier output cannot be None because we use it later in a less than statement

    def ray_cast(self, list_of_objects, camera):
        smallest_multiplier = 1e6
        closest_intersect = None
        new_ray_cast = {}
        for cuboid in list_of_objects:
            for plane_dict in cuboid.face_planes.values():
                si, intersect = self.compute_face_intersect(plane_dict,
                                                            camera)  # Checks if ray intersects a face, returns None if no intersect
                if si < smallest_multiplier:
                    smallest_multiplier = si
                    closest_intersect = intersect
                    new_ray_cast['segment'] = plane_dict['segment']
                    new_ray_cast['cuboid'] = cuboid

        if closest_intersect is None:
            smallest_multiplier = None
        else:  # if there is a valid intersection then change the cuboid's visibility status to 'visible'
            new_ray_cast['cuboid'].visibility = 'visible'  # This is the important bit, updating the visibility of the
            # cuboid that had the smallest multiplier associated with it (ie. nearest intersect)
        return smallest_multiplier, closest_intersect, new_ray_cast


# Along each obstructing segment of each object, construct rays from the camera to a certain number of points along
# the obstructing segment
def create_rays_for_3d(segment1, cuboid1, camera1, num_of_rays):
    list_of_rays = []
    # for cuboid in list_of_cuboids:
    #     for segment in cuboid.obstructing_segments:

    point1 = np.array([segment1.source().x(), segment1.source().y(), cuboid1.height + 0.01])
    point2 = np.array([segment1.target().x(), segment1.target().y(), cuboid1.height + 0.01])
    l1 = np.linspace(0, 1, num_of_rays)

    ray_points = point1 + (point2 - point1) * l1[:, None]

    for point in ray_points:
        direction = point - camera1.pos
        direction = direction.astype(float)
        normalized_direction = direction / np.linalg.norm(direction)

        angle = calculate_angle_to_y_axis(normalized_direction[0:2])
        if abs(angle) <= camera1.fov / 2:
            list_of_rays.append(Ray(camera1, normalized_direction, point))
    return list_of_rays


def create_arrangement(list_of_objects, camera):
    # Initialise the arrangement
    arr = arrangement.Arrangement()

    # Add the camera's viewing boundaries to the arrangement
    for s in camera.viewing_region_segments:
        arr.insert(s)

    # Add the cuboids' boundaries
    for cuboid in list_of_objects:
        for s in cuboid.segments_2d:
            arr.insert(s)

    return arr


def calculate_angle_to_y_axis(vector):
    unit_vector = vector / np.linalg.norm(vector)
    # Unit vector for the y axis
    y = np.array([0, 1])
    dot_product = np.dot(unit_vector, y)
    angle = np.arccos(dot_product)

    return angle


def multiple_ray_cast(list_of_rays, objects_to_check_3d, camera, ax2):
    for ray in list_of_rays:
        # for cuboid in objects_to_check_3d:
        multiplier, intercept, next_ray_cast = ray.ray_cast(objects_to_check_3d, camera)
        if multiplier is not None:
            ray.draw_ray(ax2, multiplier=multiplier)
        if next_ray_cast:
            rays = create_rays_for_3d(next_ray_cast['segment'], next_ray_cast['cuboid'], camera, 7)
            multiple_ray_cast(rays, objects_to_check_3d, camera, ax2)


def check_empty_positions_in_specific_row(empties_in_row, list_of_objects, camera1):
    arrangement = create_arrangement(list_of_objects, camera1)
    positions_to_check_visibility = []

    # Create cuboid object for each empty position in that row
    for pos in empties_in_row:
        positions_to_check_visibility.append(Cuboid(pos=pos.numpy(), size='Empty', side_length=0.3))
        positions_to_check_visibility[-1].compute_2d_segments()

    # Add the boundaries of the empty positions
    for square in positions_to_check_visibility:
        for s in square.segments_2d:
            arrangement.insert(s)

    # Initialise the rotational sweep
    vs1 = RotationalSweepVisibility(arrangement)
    q1 = Point2(camera1.pos[0], camera1.pos[1])

    # Compute the visibility surface
    face1 = arrangement.find(q1)
    vx1 = vs1.compute_visibility(q1, face1)

    visible_segments1 = []
    for v in vx1.halfedges:
        if v.curve() not in visible_segments1:  # Make sure visible_segments doesn't contain any duplicates
            visible_segments1.append(v.curve())

    for square in positions_to_check_visibility:
        # Check whether each object is visible and store in each object the segments of that object that are obstructing
        square.check_visibility(visible_segments1)

    return positions_to_check_visibility


def check_empty_positions(list_of_empty_positions, list_of_objects, camera1):
    rows = range(1, dims[1] + 1)
    empty_objects = []

    for row in rows:
        empties_in_specific_row = []
        for pos in list_of_empty_positions:
            if pos[0] == row:
                empties_in_specific_row.append(pos)
        empty_objects.extend(check_empty_positions_in_specific_row(empties_in_specific_row, list_of_objects, camera1))

    return empty_objects


def compute_state_perception(state_variables, list_of_objects, list_of_empties):
    keys = list(state_variables.keys())
    state_perception = {f"'{el}'": -1 for el in keys}
    for cuboid in list_of_objects:
        if cuboid.visibility == 'visible':
            state_perception[f"'{cuboid.pos}'"] = cuboid.size

    for empty in list_of_empties:
        if empty.visibility == 'visible':
            state_perception[f"'{empty.pos}'"] = empty.size

    for key, value in state_perception.items():
        if value == 'S':
            state_perception[key] = 1  # pyro.sample('key', Delta(tensor(1)))
        elif value == 'M':
            state_perception[key] = 2
        elif value == 'L':
            state_perception[key] = 3
        elif value == 'Empty':
            state_perception[key] = 0
    return state_perception


# Inputs: dims, sizes of objects, camera position, range and fov,
def model_3d(dims, camera_specs):
    # large = object_sizes['Large']
    # medium = object_sizes['Medium']
    # small = object_sizes['Small']
    # Load in objects
    # dims = [3, 3]

    # Choose height of objects
    # large = 10
    # medium = 2
    # small = 1

    # Initialise the camera
    # camera = Camera(np.array([1, -1, 0]), optic_range=7, fov=np.pi / 2.9)
    camera = Camera(camera_specs['pos'], optic_range=camera_specs['optic_range'], fov=camera_specs['fov'])
    camera.compute_viewing_region(num_sides=15)

    object_positions, state_variables = generate_object_positions.model(dims)
    # object_positions = {'Large positions': [tensor([2, 2])], 'Medium positions': [tensor([1, 2])],
    #                     'Small positions': [tensor([1, 1])]}
    # object_positions = {'Large positions': [tensor([3, 3]), tensor([1, 3])], 'Medium positions': [tensor([2, 1]), tensor([2, 2]), tensor([1, 2])],
    #                     'Small positions': [tensor([1, 1])]}
    # object_positions = {'Large positions': [tensor([3, 3]), tensor([3, 1])],
    #                     'Medium positions': [tensor([2, 1]), tensor([2, 2])],
    #                     'Small positions': [tensor([1, 1]), tensor([1, 2]), tensor([3, 2])],
    #                     'Empty positions': [tensor([1, 3]), tensor([2, 3])]}

    # object_positions = {'Large positions': [tensor([3, 3])],
    #                     'Medium positions': [tensor([1, 3]), tensor([2, 1]), tensor([2, 2]), tensor([2, 3])],
    #                     'Small positions': [tensor([1, 2]), tensor([3, 1]), tensor([3, 2])],
    #                     'Empty positions': [tensor([1, 1])]
    #                     }
    # object_positions = {'Large positions': [tensor([3, 1]), tensor([1, 2]), tensor([2, 2]), tensor([3, 3])],
    #                     'Medium positions': [tensor([2, 3])],
    #                     'Small positions': [tensor([3, 2])],
    #                     'Empty positions': [tensor([1, 1]), tensor([1, 3]), tensor([2, 1])]
    #                     }
    # object_positions = {'Large positions': [tensor([4, 2]), tensor([4, 3])],
    #                     'Medium positions': [tensor([2, 3]), tensor([2, 2]), tensor([1, 2]), tensor([4, 1])],
    #                     'Small positions': [tensor([1, 1])],
    #                     'Empty positions': [tensor([1, 3]), tensor([2, 1])]
    #                     }
    # object_positions = {'Large positions': [tensor([3, 3])],
    #                     'Medium positions': [tensor([1, 2])],
    #                     'Small positions': [],
    #                     'Empty positions': [tensor([1, 1]), tensor([1, 2]), tensor([1, 3]), tensor([2, 1]), tensor([2, 3]), tensor([3, 1]), tensor([3, 2])]
    #                     }

    # Initialise cuboids
    objects = []
    for position in object_positions['Large positions']:
        objects.append(Cuboid(pos=position.numpy(), size='L', side_length=0.3))

    for position in object_positions['Medium positions']:
        objects.append(Cuboid(pos=position.numpy(), size='M', side_length=0.3))

    for position in object_positions['Small positions']:
        objects.append(Cuboid(pos=position.numpy(), size='S', side_length=0.3))

    # Compute the 2d segments for each cuboid
    for cuboid in objects:
        cuboid.compute_2d_segments()

    # Create first figure
    fig = plt.figure(1, figsize=(6, 6))
    ax = fig.add_subplot(111)

    #### 2D visibility

    # Check empty positions
    empty_objects = check_empty_positions(object_positions['Empty positions'], objects, camera)

    # Create the 2D arrangement
    arr = create_arrangement(objects, camera)

    # Draw the arrangement each object with different colour, this is a mess.
    for he in arr.halfedges:  # each edge is separated into two half edges
        for cuboid in objects:
            for i in cuboid.segments_2d:
                if i.has_on(he.curve().target()) and i.has_on(he.curve().source()):
                    draw.draw(he.curve(), color=cuboid.color, marker='')  # curve() turns the half-edges into segments?
                    break
            else:
                continue
            break
        else:
            draw.draw(he.curve(), color="#1f77b4", marker='')

    # Initialise the rotational sweep
    vs = RotationalSweepVisibility(arr)
    q = Point2(camera.pos[0], camera.pos[1])

    # Compute the visibility surface
    face = arr.find(q)
    vx = vs.compute_visibility(q, face)

    # Create a list containing all the visible segments
    visible_segments = []
    for v in vx.halfedges:
        if v.curve() not in visible_segments:  # Make sure visible_segments doesn't contain any duplicates
            visible_segments.append(v.curve())
        draw.draw(v.curve(), color='red', visible_point=False)

    # Draw the end points of the segments that are obstructing the camera's view
    # for point in vx.vertices:
    #     draw.draw(point.point(), marker='x')

    # Check whether each object is visible and store in each object the segments of that object that are obstructing
    for cuboid in objects:
        cuboid.check_visibility(visible_segments)

    # Draw the camera origin in 2D
    draw.draw(q, color='magenta')

    #### 3D visibility
    # Initialise second figure (3D)
    fig2 = plt.figure(2, figsize=(6, 6))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plot each cuboid
    for cuboid in objects:
        cuboid.plot_cuboid(ax2, alpha=1)

    # Plot the camera
    camera.plot_camera(ax2)

    # Create the list of rays for the simplified 3D ray casting
    list_of_rays = []
    for cuboid in objects:
        for segment in cuboid.obstructing_segments:
            list_of_rays.extend(create_rays_for_3d(segment, cuboid, camera, 7))

    # Compute the plane equations for each face of each cuboid to be used in the ray casting
    for cuboid in objects:
        cuboid.compute_plane_equations()

    # Get a list of all the objects that aren't visible in 2D to check if they are visible in 3D
    objects_to_check_3d = []
    for cuboid in objects:
        if cuboid.visibility == 'not visible':
            objects_to_check_3d.append(cuboid)

    # Draw the rays that intersect an object not visible in the 2d world
    multiple_ray_cast(list_of_rays, objects_to_check_3d, camera, ax2)

    # Specify ax2 labels, dimensions and ticks
    ax2.set_xlabel('X')
    ax2.set_xlim3d(-1, 4)

    ax2.set_ylabel('Y')
    ax2.set_ylim3d(-1, 4)

    ax2.set_zlabel('Z')
    ax2.set_zlim3d(0, 10)

    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(1))

    # Calculate the robot's perception of the state
    state_perception = compute_state_perception(state_variables, objects, empty_objects)

    # Combine the complete knowledge of the state and the robot's perception of the state into output dictionary
    output = {**state_variables, **state_perception}

    return output


dims = [3, 3]
num_samples = []
# Choose height of objects
large = 8
medium = 3
small = 1

camera_specs = {'pos': np.array([1, -1, 0]),
                'optic_range': 9,
                'fov': np.pi * 0.7
                }
output = model_3d(dims, camera_specs)
print(output)
plt.show()
