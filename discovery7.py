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
from pgmpy.estimators import BDeuScore, BicScore, HillClimbSearch, MmhcEstimator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from cdt.causality.graph.model import GraphModel
from cdt.causality.graph.bnlearn import BNlearnAlgorithm
from pgmpy.estimators.base import StructureEstimator
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import gsq


def position_nodes(variables):
    state_variables = variables[:int(len(variables) / 2)]
    perception_variables = variables[int(len(variables) / 2):]

    fixed_nodes = {}
    y_coords = np.linspace(1, -1, len(state_variables))
    for idx, val in enumerate(state_variables):
        pos = np.array([0, y_coords[idx]])
        fixed_nodes[val] = pos

    theta = np.linspace(1, -1, len(perception_variables)) * (1 / 5 * np.pi)
    for idx, val in enumerate(perception_variables):
        pos = np.array([y_coords[idx] / np.tan(theta[idx]), y_coords[idx]])
        fixed_nodes[val] = pos

    return fixed_nodes


def create_ground_truth(dims, variables):
    """
    Creates the ground truth DAG
    :param dims: list containing the dimensions of the grid
    :param variables:
    :return: networkx DiGraph of the ground truth
    """
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
        edges[cause] = [node for node in state_variables if
                        torch.equal(cause, node) or (torch.eq(cause[1], node[1]) and torch.gt(node[0], cause[0]))]
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


def calculate_accuracy(predicted_graph, ground_truth):
    """
    Calculates the accuracy of the predicted graph compared to the ground truth by treating it like a classification
    problem with two classes: edge or no edge

    :param predicted_graph: networkx DiGraph that is outputted by the discovery algorithm
    :param ground_truth: the ground truth networkx DiGraph
    :return: the accuracy of the predicted graph
    """

    true_labels = cdt.metrics.retrieve_adjacency_matrix(ground_truth)
    pred = cdt.metrics.retrieve_adjacency_matrix(predicted_graph, order_nodes=ground_truth.nodes() if isinstance(ground_truth, nx.DiGraph) else None)

    accuracy = accuracy_score(true_labels.ravel(), pred.ravel())

    return accuracy


def calculate_f1(predicted_graph, ground_truth):
    """
    Calculates the f1 score of the predicted graph compared to the ground truth by treating it like a classification
    problem with two classes: edge or no edge

    :param predicted_graph: networkx DiGraph that is outputted by the discovery algorithm
    :param ground_truth: the ground truth networkx DiGraph
    :return: the f1 score of the predicted graph
    """

    true_labels = cdt.metrics.retrieve_adjacency_matrix(ground_truth)
    pred = cdt.metrics.retrieve_adjacency_matrix(predicted_graph, order_nodes=ground_truth.nodes() if isinstance(ground_truth, nx.DiGraph) else None)

    f1 = f1_score(true_labels.ravel(), pred.ravel())

    return f1


def calculate_roc_auc(predicted_graph, ground_truth):
    """
    Calculates the area under the ROC curve of the predicted graph compared to the ground truth by treating it like a
    classification problem with two classes: edge or no edge

    :param predicted_graph: networkx DiGraph that is outputted by the discovery algorithm
    :param ground_truth: the ground truth networkx DiGraph
    :return: the area under the ROC curve of the predicted graph
    """

    true_labels = cdt.metrics.retrieve_adjacency_matrix(ground_truth)
    pred = cdt.metrics.retrieve_adjacency_matrix(predicted_graph, order_nodes=ground_truth.nodes() if isinstance(ground_truth, nx.DiGraph) else None)

    roc_auc = roc_auc_score(true_labels.ravel(), pred.ravel())

    return roc_auc


def calculate_pr_auc(predicted_graph, ground_truth):
    """
        Calculates the area under the precision-recall curve of the predicted graph compared to the ground truth by
        treating it like a classification problem with two classes: edge or no edge

        :param predicted_graph: networkx DiGraph that is outputted by the discovery algorithm
        :param ground_truth: the ground truth networkx DiGraph
        :return: the area under the PR curve of the predicted graph
        """

    true_labels = cdt.metrics.retrieve_adjacency_matrix(ground_truth)
    pred = cdt.metrics.retrieve_adjacency_matrix(predicted_graph,
                                                 order_nodes=ground_truth.nodes() if isinstance(ground_truth,
                                                                                                nx.DiGraph) else None)

    pr_auc = average_precision_score(true_labels.ravel(), pred.ravel())

    return pr_auc


# Let cdt know the path to access R for the necessary R packages
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.1.2/bin/RScript.exe'

cdt.SETTINGS.GPU = 1

# Load in the data
data_file = '2x2_100_samples1.csv'
data = pd.read_csv(data_file)
bn_learn_data = data.replace(to_replace={-1: -1.1, 0: 0.1, 1: 1.1, 2: 2.1, 3: 3.1})


# Create list of objects: Vary significance levels and compare
algorithms = []
# CDT Algorithms
algorithms.append(PC(CItest='discrete'))
# algorithms.append(cdt.causality.graph.GES(score='int'))  # Find a way to output the BIC score as well
# algorithms.append(cdt.causality.graph.CGNN(nruns=1, gpus=1))  # Takes too long, over 5 mins, didn't wait until the end
# algorithms.append(cdt.causality.graph.GIES())
# algorithms.append(cdt.causality.graph.CCDr())
# algorithms.append(cdt.causality.graph.LiNGAM())  # Error! - system is computationally singular

# BNLearn Algorithms
# algorithms.append(cdt.causality.graph.bnlearn.GS())  # Error! - bnlearn integer issue
# algorithms.append(cdt.causality.graph.bnlearn.MMPC())  # Error! - same as above
# algorithms.append(cdt.causality.graph.bnlearn.IAMB())
# algorithms.append(cdt.causality.graph.bnlearn.Fast_IAMB())  # Error! - same as above
# algorithms.append(cdt.causality.graph.bnlearn.Inter_IAMB())
# algorithms.append(cdt.causality.graph.SAMv1()) # Error! - long error figure out later, infinite runtime on docker

# PGMPY algorithms

# algorithms.append(HillClimbSearch(data))
# algorithms.append(MmhcEstimator(data))  # Takes a long time, worth putting in a max computation time

# Can't get the FCI Algorithm to work:(


# Fix the state variables' positions and the perception variables' starting points
variables = data.columns
fixed_nodes = position_nodes(variables)

# Generate the ground truth DAG
dims = [2, 2]
ground_truth = create_ground_truth(dims, variables)

# Create the scoring objects
bdeu = BDeuScore(data, equivalent_sample_size=5)
bic = BicScore(data)

results = pd.DataFrame()

for idx, value in enumerate(algorithms):
    result = {}

    if isinstance(value, BNlearnAlgorithm):
        start = time.time()
        output = value.predict(bn_learn_data)
        end = time.time()
    elif isinstance(value, GraphModel):
        start = time.time()
        output = value.predict(data)
        end = time.time()
    elif isinstance(value, StructureEstimator):
        start = time.time()
        output = value.estimate()
        end = time.time()
    else:
        start = time.time()
        output = fci(data, independence_test_method=gsq)
        end = time.time()


    # Metrics
    result['Algorithm'] = value
    result['Duration'] = end - start
    result['SHD'] = cdt.metrics.SHD(ground_truth, output)
    result['SID'] = cdt.metrics.SID(ground_truth, output)
    result['PR AUC'] = cdt.metrics.precision_recall(ground_truth, output)[0]
    # result['PR AUC2'] = calculate_pr_auc(output, ground_truth)
    result['ROC AUC'] = calculate_roc_auc(output, ground_truth)
    result['F1'] = calculate_f1(output, ground_truth)
    result['Accuracy'] = calculate_accuracy(output, ground_truth)
    result['BDeu'] = bdeu.score(output)
    result['BIC'] = bic.score(output)

    results = results.append(result, ignore_index=True)


# fig1, ax1 = plt.subplots()
# ax1.set_title('PC Algorithm')
# nx.draw(output, pos=fixed_nodes, with_labels=True)

plt.show()
pd.set_option('display.max_columns', None)
print(results)

results.to_csv(f'Results_{data_file}')
