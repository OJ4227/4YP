import pandas as pd
import networkx as nx
from cdt.causality.graph import PC
import matplotlib.pyplot as plt
import cdt
import numpy as np
from networkx import DiGraph
import torch
from torch import tensor
import time



def position_nodes(variables):

    state_variables = variables[:int(len(variables)/2)]
    perception_variables = variables[int(len(variables)/2):]
    # print(state_variables)
    # print(perception_variables)
    fixed_nodes = {}
    y_coords = np.linspace(1, -1, len(state_variables))
    for idx, val in enumerate(state_variables):
        pos = np.array([0, y_coords[idx]])
        fixed_nodes[val] = pos

    theta = np.linspace(1, -1, len(perception_variables)) * (1/5 * np.pi)
    for idx, val in enumerate(perception_variables):
        pos = np.array([y_coords[idx]/np.tan(theta[idx]), y_coords[idx]])
        fixed_nodes[val] = pos

    return fixed_nodes


def create_ground_truth(dims, variables):
    # Separate variables into the state variables and the perception of the state variables
    state_variables_str = variables[:int(len(variables) / 2)]
    state_perception_str = variables[int(len(variables) / 2):]

    # Create a list containing the state variables
    rows = dims[0]
    columns = dims[1]
    state_variables = []
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            state_variables.append(tensor([row, column]))  # Label positions by row and column

    # Initialise the graph with the state and perceived state variables as nodes
    ground_truth = DiGraph()
    # for var in state_variables_str:
    #     ground_truth.add_node(node=var)
    #
    # for var in state_perception_str:
    #     ground_truth.add_node(node=var)
    ground_truth.add_nodes_from(variables)

    # edges = create a dictionary with the cause as the key, the values are lists of nodes that are the effects,
    # convert the effects to strings - you need to keep them all as tensors to compare the columns and rows and then
    # convert to strings later
    edges = {}
    for cause in state_variables:
        edges[cause] = [node for node in state_variables if torch.equal(cause, node) or (torch.eq(cause[1], node[1]) and torch.greater(node[0], cause[0]))]
    # edges contains the node that is the cause and a list of nodes that are caused by it, but the affected nodes are
    # still in tensor form and so look like state variables

    # Now convert the affected nodes to strings with apostrophes to identify them as perception variables
    # Initialise edges_input - our dictionary that we will use to add edges to the DAG
    edges_input = {k: [] for k, v in edges.items()}
    # Add in the string versions of the affected nodes
    [edges_input[k].append(f"'{str(i.numpy())}'") for k, v in edges.items() for i in v]
    # Convert the keys to strings
    edges_input = {str(k.numpy()): v for k, v in edges_input.items()}

    # Add the edges to the DAG
    for k, v in edges_input.items():
        for node in v:
            ground_truth.add_edge(k, node)

    return ground_truth


# Let cdt know the path to access R for the necessary R packages
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.1.2/bin/RScript.exe'

print(cdt.utils.R.RPackages)
print(cdt.RPackages.check_R_package('SID'))
# print(cdt.RPackages.SID)


# Load in the data
data = pd.read_csv('num_robot_data_state_perception2_2x2_100_samples.csv')
# print(data)
# data = data.rename(columns={'[1 1]': 1, '[1 2]': 2, '[2 1]':3, '[2 2]':4, "'[1 1]'": 5, "'[1 2]'": 6, "'[2 1]'": 7, "'[2 2]'": 8})
# print(data)

# Apply the PC algorithm
obj = PC(CItest='discrete')
# obj = cdt.causality.graph.GES(score='int')  # Find a way to output the BIC score as well
# obj = cdt.causality.graph.CGNN()  # Takes too long, over 5 mins, didn't wait until the end
# obj = cdt.causality.graph.GIES()
# obj = cdt.causality.graph.CCDr()
# obj = cdt.causality.graph.LiNGAM()  # Error! - system is computationally singular
# obj = cdt.causality.graph.GS()  # Error! - bnlearn integer issue
# obj = cdt.causality.graph.bnlearn.MMPC()  # Error! - same as above
# obj = cdt.causality.graph.bnlearn.Fast_IAMB()  # Error! - same as above
# obj = cdt.causality.graph.SAMv1() # Error! - long error figure out later
cdt.causality.graph.bnlearn.BNlearnAlgorithm

start = time.time()
output = obj.predict(data)
end = time.time()
duration = end - start

# Fix the state variables' positions and the perception variables' starting points
variables = data.columns
fixed_nodes = position_nodes(variables)

# Generate the ground truth DAG
dims = [2, 2]
ground_truth = create_ground_truth(dims, variables)

# nx.draw_networkx(output, pos=nx.spring_layout(output, pos=fixed_nodes, fixed=perception_variables), font_size=8)
# nx.draw_networkx(output, pos=fixed_nodes, font_size=8)
# nx.draw_networkx(output, font_size=8)

ground_truth_adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(ground_truth)

fig1, ax1 = plt.subplots()
ax1.set_title('PC algorithm')
nx.draw(output, pos=fixed_nodes, with_labels=True)

fig2, ax2 = plt.subplots()
ax2.set_title('Ground Truth')
nx.draw(ground_truth, pos=fixed_nodes, with_labels=True)


# Metrics
SHD = cdt.metrics.SHD(ground_truth, output)
# SID = SID(ground_truth, output)
SID = cdt.metrics.SID(ground_truth, output)
precision_recall_area = cdt.metrics.precision_recall(ground_truth, output)[0]
# cdt.metrics.precision_recall_curve()

print(f'The SHD is {SHD}. The area under the precision recall curve is {precision_recall_area}. Time taken is {duration}s')

plt.show()
