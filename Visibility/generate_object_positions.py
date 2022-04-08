import torch
from pyro import sample
import pyro
import pyro.distributions as dist
from torch import tensor
import pandas as pd
import numpy as np
import os


# Variables are now grid positions
# Output is the state and perception of the grid positions - None=-1, Empty=0, S=1, M=2, L=3
# Now there is no limit on objects in the grid, each grid samples from a categorical distribution to see what object is
# in it (L, M, S, Empty)

# Assuming you can't see a large behind a large, medium behind a medium and a small behind a small, using two
# large, two medium and two small objects


class Discrete_Uniform(dist.TorchDistribution):
    arg_constraints = {}

    def __init__(self, vals):
        self.vals = vals
        probs = torch.ones(len(vals))
        self.categorical = dist.Categorical(probs)
        super(Discrete_Uniform, self).__init__(self.categorical.batch_shape,
                                               self.categorical.event_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.vals[self.categorical.sample(sample_shape)]

    def log_prob(self, value):
        idx = (self.vals == value).nonzero()
        return self.categorical.log_prob(idx)


def df_tensor_to_value(df):
    """
    Converts a data frame of tensors to a data frame of numpy arrays

    :param df: a data frame containing only tensor objects
    :return: None
    """
    for column in df.columns:
        for row in range(len(df.index)):
            if df.loc[row][column] is not None and not isinstance(df.loc[row][column], str):
                df.loc[row][column] = [item.numpy() for item in df.loc[row][column]]


def check_larges(object_positions):
    """
    Tells you which large objects are visible by checking which large objects are not visible as they are behind another
    large object

    :param object_positions: dictionary containing the positions of the large, medium and small objects in the grid
    :return vis_large: a list containing all of the visible large objects
    """
    large_positions = object_positions['Large positions']
    larges_not_visible = []

    # Add to the list the large objects that are behind other large objects and therefore not visible

    for large_pos in large_positions:
        for large_pos2 in large_positions:
            if torch.eq(large_pos[1], large_pos2[1]) and torch.greater(large_pos[0], large_pos2[0]):
                if not any(torch.equal(large_pos, k) for k in larges_not_visible):  # Check that this large hasn't
                    # already been included in the set of not visible mediums
                    larges_not_visible.append(large_pos)  # this list contains all the smalls that are not visible

    # If any of the large positions are also included in the list of larges that are not visible then move on. If this
    # large position isn't included in the list of not visible larges then it must be visible and therefore append to
    # list of visible larges
    vis_large = []
    for pos in large_positions:
        if not any(torch.equal(pos, i) for i in larges_not_visible):
            vis_large.append(pos)

    return vis_large


def check_mediums(object_positions):
    """
    Tells you which medium objects are visible by checking which medium objects are not visible as they are behind a
    large or medium object

    :param object_positions: dictionary containing the positions of the large, medium and small objects in the grid
    :return: vis_medium: a list containing all of the visible medium objects
    """
    large_positions = object_positions['Large positions']
    medium_positions = object_positions['Medium positions']
    # Create a list containing all the mediums in the same column as a large and that have a higher row
    mediums_not_visible = []
    for med_pos in medium_positions:
        for large_pos in large_positions:
            if torch.eq(med_pos[1], large_pos[1]) and torch.greater(med_pos[0], large_pos[0]):
                if not any(torch.equal(med_pos, k) for k in mediums_not_visible):  # Check that this medium hasn't
                    # already been included in the set of not visible mediums
                    mediums_not_visible.append(med_pos)  # this list will contain all the mediums that are not visible

        # Add to the list the medium objects that are behind other medium objects and therefore not visible
        for med_pos2 in medium_positions:
            if torch.eq(med_pos[1], med_pos2[1]) and torch.greater(med_pos[0], med_pos2[0]):
                if not any(torch.equal(med_pos, k) for k in mediums_not_visible):  # Check that this medium hasn't
                    # already been included in the set of not visible mediums
                    mediums_not_visible.append(med_pos)  # this list contains all the mediums that are not visible

    # If any of the medium positions are also included in the list of mediums that are not visible then move on. If this
    # medium position isn't included in the list of not visible mediums then it must be visible and therefore append to
    # list of visible mediums
    vis_medium = []
    for pos in medium_positions:
        if not any(torch.equal(pos, i) for i in mediums_not_visible):
            vis_medium.append(pos)

    return vis_medium


def check_smalls(object_positions):
    """
    Tells you which small objects are visible by checking which small objects are not visible as they are behind a
    large, medium or small object

    :param object_positions: dictionary containing the positions of the large, medium and small objects in the grid
    :return: vis_small: a list containing all of the visible small objects
    """
    large_positions = object_positions['Large positions']
    medium_positions = object_positions['Medium positions']
    small_positions = object_positions['Small positions']

    # Create a list containing all the smalls in the same column as a small, medium or large and that have a higher row

    smalls_not_visible = []

    for small_pos in small_positions:
        for large_pos in large_positions:
            if torch.eq(small_pos[1], large_pos[1]) and torch.greater(small_pos[0], large_pos[0]):
                if not any(torch.equal(small_pos, k) for k in smalls_not_visible):  # Check that this small hasn't
                    # already been included in the set of not visible smalls
                    smalls_not_visible.append(small_pos)  # this list will contain all the smalls that are not visible

        for med_pos in medium_positions:
            if torch.eq(small_pos[1], med_pos[1]) and torch.greater(small_pos[0], med_pos[0]):
                if not any(torch.equal(small_pos, k) for k in smalls_not_visible):  # Check that this small hasn't
                    # already been included in the set of not visible smalls
                    smalls_not_visible.append(small_pos)  # this list will contain all the smalls that are not visible

        for small_pos2 in small_positions:
            if torch.eq(small_pos[1], small_pos2[1]) and torch.greater(small_pos[0], small_pos2[0]):
                if not any(torch.equal(small_pos, k) for k in smalls_not_visible):  # Check that this small hasn't
                    # already been included in the set of not visible smalls
                    smalls_not_visible.append(small_pos)  # this list contains all the smalls that are not visible

    # Create a list of all the objects included in the small_positions and the not in the visible smalls ie. the small
    # objects that are visible

    vis_small = []
    for pos in small_positions:
        if not any(torch.equal(pos, i) for i in smalls_not_visible):
            vis_small.append(pos)

    return vis_small


def output_perception(grid_positions, object_positions):
    """
    Outputs the perceived state of the system. Objects behind larger ones or ones of the same size are hidden and empty
    grid positions behind any object are hidden. If an object is visible, its position is known.

    :param grid_positions: list of 2D tensors corresponding to grid positions
    :param object_positions: dictionary containing the positions of the large, medium and small objects in the grid

    :return: state_perception: a dictionary where the keys are the string versions of the grid positions and the values
    are either 3 ('L'), 2 ('M'), 1 ('S'), 0 ('Empty') or -1 (Hidden objects/grid positions)
    """

    large_positions = object_positions['Large positions']
    medium_positions = object_positions['Medium positions']
    small_positions = object_positions['Small positions']

    # Convert lists of tensors to lists of strings so we can search through the lists - we can't search for mutable 2D
    # objects
    str_grid_positions = [str(i.numpy()) for i in grid_positions]

    large_positions1 = [str(i.numpy()) for i in large_positions]
    medium_positions1 = [str(i.numpy()) for i in medium_positions]
    small_positions1 = [str(i.numpy()) for i in small_positions]

    # Initialise the output with a default value of 0 ('Empty')
    state_perception = {el: 0 for el in str_grid_positions}

    # Find the position of the object in each column that has the lowest row, so we can find which empty grid positions
    # behind it aren't visible

    columns = grid_positions[-1][1].item()
    object_lowest_row_in_column = np.array([1000 for i in range(columns)])  # Choose 1000 as an arbitrary large number

    objects = large_positions + medium_positions + small_positions

    for pos in objects:
        for col in range(1, columns + 1):
            if torch.eq(pos[1], col):
                if pos[0] < object_lowest_row_in_column[col - 1]:
                    object_lowest_row_in_column[col - 1] = pos[0]

    # Determine which grid positions are empty
    grids_empty = []

    for pos in grid_positions:
        if not any((torch.equal(pos, i) for i in large_positions)) \
                and not any((torch.equal(pos, j) for j in medium_positions)) \
                and not any((torch.equal(pos, k) for k in small_positions)):
            grids_empty.append(pos)

    # Determine which of the empty grid positions are behind the object in that column with the lowest row and are
    # therefore not visible
    grids_not_visible = []

    for pos in grids_empty:
        if torch.greater(pos[0], object_lowest_row_in_column[pos[1] - 1]):
            grids_not_visible.append(pos)

    # Determine which objects are visible
    vis_large = check_larges(object_positions)
    vis_med = check_mediums(object_positions)
    vis_small = check_smalls(object_positions)

    # Convert these lists of tensors to lists of strings
    vis_large1 = [str(i.numpy()) for i in vis_large]
    vis_med1 = [str(i.numpy()) for i in vis_med]
    vis_small1 = [str(i.numpy()) for i in vis_small]
    grids_not_visible1 = [str(i.numpy()) for i in grids_not_visible]

    # Check whether each grid position contains a visible large/medium/small, a non-visible empty grid, or a non-visible
    # large/medium/small and set value accordingly

    for position in state_perception:
        if position in vis_large1:
            state_perception[position] = 3
        elif position in vis_med1:
            state_perception[position] = 2
        elif position in vis_small1:
            state_perception[position] = 1
        elif position in grids_not_visible1:
            state_perception[position] = -1
        elif position in large_positions1 and state_perception[position] == 0:
            state_perception[position] = -1
        elif position in medium_positions1 and state_perception[position] == 0:
            state_perception[position] = -1
        elif position in small_positions1 and state_perception[position] == 0:
            state_perception[position] = -1

    return state_perception


def model(dims):
    """
    Creates a grid of dimension 'dims' and populates it with objects sampled from a discrete uniform distribution of
    large, medium, small and empty. It then returns the state of the grid and the perception of the grid in a single
    dictionary.

    :param dims: List containing the dimensions of the grid [rows, columns]
    :return output: Dictionary containing the state with strings of the grid positions as keys eg. '[1, 3]' and the
    perception of the state with strings of the grid positions and an apostrophe as keys eg. '[1, 3]''
    """

    rows = dims[0]
    columns = dims[1]
    grid_positions = []
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            grid_positions.append(tensor([row, column]))  # Label positions by row and column

    # Assign object positions by sampling from a discrete uniform dist containing the position indices of the remaining
    # vals/positions and remove the selected val

    # Initialise a dictionary to contain all the object positions
    object_positions = {
        'Large positions': [],
        'Medium positions': [],
        'Small positions': [],
        'Empty positions': []
    }

    possible_objects = [tensor(0), tensor(1), tensor(2), tensor(3)]  # Empty, Small, Medium and Large
    state_variables = {}
    for idx, pos in enumerate(grid_positions):
        # obj = sample('obj', Discrete_Uniform(possible_objects))

        # state_variables[f'{pos.numpy()}'] = sample(f'{pos.numpy()}', dist.Delta(obj))
        state_variables[f'{pos.numpy()}'] = sample(f'{pos.numpy()}', Discrete_Uniform(possible_objects))
        if torch.eq(state_variables[f'{pos.numpy()}'], tensor(1)):
            object_positions['Small positions'].append(pos)
        elif torch.eq(state_variables[f'{pos.numpy()}'], tensor(2)):
            object_positions['Medium positions'].append(pos)
        elif torch.eq(state_variables[f'{pos.numpy()}'], tensor(3)):
            object_positions['Large positions'].append(pos)
        elif torch.eq(state_variables[f'{pos.numpy()}'], tensor(0)):
            object_positions['Empty positions'].append(pos)

    for key in state_variables:
        state_variables[key] = state_variables[key].item()

    return object_positions, state_variables

# object_positions, state_variables = model([2, 2])
# print(object_positions)
# print(state_variables)

# intervened_model = pyro.poutine.do(model, data={'[1 1]': tensor(1)})
# object_positions, state_variables = intervened_model([2, 2])
# print(object_positions)
# print(state_variables)
