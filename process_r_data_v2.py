import pandas as pd
import numpy as np
import cdt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import torch
from torch import tensor
from networkx import DiGraph
from matplotlib import pyplot as plt
import networkx as nx
from pgmpy.estimators import BDeuScore, BicScore
import os
from pgmpy.base import DAG
from pgmpy.metrics.metrics import structure_score

# This script combines the time data with the results that we already calculated (which took multiple days of computation)


def create_ground_truth_adj(dims, variables):
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

    true_labels = cdt.metrics.retrieve_adjacency_matrix(ground_truth)
    return ground_truth


def position_nodes(variables):
    state_variables = variables[:int(len(variables) / 2)]
    perception_variables = variables[int(len(variables) / 2):]

    fixed_nodes = {}
    theta = np.linspace(1, -1, len(perception_variables)) * (1 / 5 * np.pi)
    y_coords = np.linspace(1, -1, len(state_variables))
    for idx, val in enumerate(state_variables):
        pos = np.array([-y_coords[idx] / np.tan(theta[idx]), y_coords[idx]])
        pos[np.isnan(pos)] = -1.6
        fixed_nodes[val] = pos

    theta = np.linspace(1, -1, len(perception_variables)) * (1 / 5 * np.pi)
    for idx, val in enumerate(perception_variables):
        pos = np.array([y_coords[idx] / np.tan(theta[idx]), y_coords[idx]])
        pos[np.isnan(pos)] = 1.6
        fixed_nodes[val] = pos

    return fixed_nodes


def get_variables(dims):
    rows = dims[0]
    columns = dims[1]
    grid_positions = []
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            grid_positions.append(tensor([row, column]))

    state_variables = [f'{i.numpy()}' for i in grid_positions]
    perception_variables = [f"'{i}'" for i in state_variables]
    variables = state_variables + perception_variables
    return variables


def position_nodes(variables):
    state_variables = variables[:int(len(variables) / 2)]
    perception_variables = variables[int(len(variables) / 2):]

    fixed_nodes = {}
    theta = np.linspace(1, -1, len(perception_variables)) * (1 / 5 * np.pi)
    y_coords = np.linspace(1, -1, len(state_variables))
    for idx, val in enumerate(state_variables):
        pos = np.array([-y_coords[idx] / np.tan(theta[idx]), y_coords[idx]])
        pos[np.isnan(pos)] = -1.6
        fixed_nodes[val] = pos

    theta = np.linspace(1, -1, len(perception_variables)) * (1 / 5 * np.pi)
    for idx, val in enumerate(perception_variables):
        pos = np.array([y_coords[idx] / np.tan(theta[idx]), y_coords[idx]])
        pos[np.isnan(pos)] = 1.6
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
    pred = cdt.metrics.retrieve_adjacency_matrix(predicted_graph,
                                                 order_nodes=ground_truth.nodes() if isinstance(ground_truth,
                                                                                                nx.DiGraph) else None)

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
    pred = cdt.metrics.retrieve_adjacency_matrix(predicted_graph,
                                                 order_nodes=ground_truth.nodes() if isinstance(ground_truth,
                                                                                                nx.DiGraph) else None)

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
    pred = cdt.metrics.retrieve_adjacency_matrix(predicted_graph,
                                                 order_nodes=ground_truth.nodes() if isinstance(ground_truth,
                                                                                                nx.DiGraph) else None)

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


num_of_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
list_of_dims = [[2, 2], [3, 3], [2, 4], [4, 2], [4, 4]]
alg_types = ['Constraint based', 'Score based', 'Hybrid', 'rmsax2']

for dims in list_of_dims:
    for num in num_of_samples:
        for i in range(1, 11):
            # for alg_type in alg_types:
            # Load in the file containing timings of the algorithms
            # load in each file in version 2 directory, combine into data frame, then insert into other data frame


            read_path = os.path.dirname(os.path.abspath(
                __file__)) + f'/data/{dims[0]}x{dims[1]}/R outputs/outputs_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/'
            constraint_files = os.listdir(read_path + 'Constraint based/Version 2')
            score_files = os.listdir(read_path + 'Score based/Version 2')
            hybrid_files = os.listdir(read_path + 'Hybrid/Version 2')
            rsmax2_files = os.listdir(read_path + 'rsmax2/Version 2')

            adj_size = dims[0] * dims[1] * 2

            ## Constraint based
            constraint_time_results = pd.DataFrame()
            for file in constraint_files:
                time_data_file = pd.read_csv(read_path + 'Constraint based/Version 2/' + file)
                time_data_file = time_data_file.drop(labels=[time_data_file.columns[0]], axis=1)

                alg_name = time_data_file.iloc[1, adj_size]
                calls = time_data_file.iloc[1, adj_size + 1]
                duration = time_data_file.iloc[1, adj_size + 2]

                result = {}

                result['Algorithm'] = alg_name
                result['No. of calls'] = calls
                result['Duration (s)'] = duration
                constraint_time_results = constraint_time_results.append(result, ignore_index=True)


            # Read in the preprocessed results so we can tack on the time results
            constraint_existing_file = f"C:/Users/order/Documents/Oxford/4th Year/4YP/Pyro practice/data/{dims[0]}x{dims[1]}/R results/results_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/constraint_based_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"
            constraint_existing_data = pd.read_csv(constraint_existing_file)
            constraint_existing_data = constraint_existing_data.drop(labels=[constraint_existing_data.columns[0]], axis=1)
            constraint_idx = constraint_existing_data.columns.get_loc('No. of calls')


            constraint_existing_data.insert(constraint_idx+1,'Duration (s)', constraint_time_results['Duration (s)'])

            ## Score based
            score_time_results = pd.DataFrame()
            for file in score_files:
                time_data_file = pd.read_csv(read_path + 'Score based/Version 2/' + file)
                time_data_file = time_data_file.drop(labels=[time_data_file.columns[0]], axis=1)

                alg_name = time_data_file.iloc[1, adj_size]
                calls = time_data_file.iloc[1, adj_size + 1]
                duration = time_data_file.iloc[1, adj_size + 2]

                result = {}

                result['Algorithm'] = alg_name
                result['No. of calls'] = calls
                result['Duration (s)'] = duration
                score_time_results = score_time_results.append(result, ignore_index=True)

            # Read in the preprocessed results so we can tack on the time results
            score_existing_file = f"C:/Users/order/Documents/Oxford/4th Year/4YP/Pyro practice/data/{dims[0]}x{dims[1]}/R results/results_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/score_based_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"
            score_existing_data = pd.read_csv(score_existing_file)
            score_existing_data = score_existing_data.drop(labels=[score_existing_data.columns[0]], axis=1)
            score_idx = score_existing_data.columns.get_loc('No. of calls')

            score_existing_data.insert(score_idx + 1, 'Duration (s)', score_time_results['Duration (s)'])

            ## Hybrid
            hybrid_time_results = pd.DataFrame()
            for file in hybrid_files:
                time_data_file = pd.read_csv(read_path + 'Hybrid/Version 2/' + file)
                time_data_file = time_data_file.drop(labels=[time_data_file.columns[0]], axis=1)

                alg_name = time_data_file.iloc[1, adj_size]
                calls = time_data_file.iloc[1, adj_size + 1]
                duration = time_data_file.iloc[1, adj_size + 2]

                result = {}

                result['Algorithm'] = alg_name
                result['No. of calls'] = calls
                result['Duration (s)'] = duration
                hybrid_time_results = hybrid_time_results.append(result, ignore_index=True)

            # Read in the preprocessed results so we can tack on the time results
            hybrid_existing_file = f"C:/Users/order/Documents/Oxford/4th Year/4YP/Pyro practice/data/{dims[0]}x{dims[1]}/R results/results_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/hybrid_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"
            hybrid_existing_data = pd.read_csv(hybrid_existing_file)
            hybrid_existing_data = hybrid_existing_data.drop(labels=[hybrid_existing_data.columns[0]], axis=1)
            hybrid_idx = hybrid_existing_data.columns.get_loc('No. of calls')

            hybrid_existing_data.insert(hybrid_idx + 1, 'Duration (s)', hybrid_time_results['Duration (s)'])

            ## rsmax2
            rsmax2_time_results = pd.DataFrame()
            for file in rsmax2_files:
                time_data_file = pd.read_csv(read_path + 'rsmax2/Version 2/' + file)
                time_data_file = time_data_file.drop(labels=[time_data_file.columns[0]], axis=1)

                alg_name = time_data_file.iloc[1, adj_size]
                calls = time_data_file.iloc[1, adj_size + 1]
                duration = time_data_file.iloc[1, adj_size + 2]
                if (alg_name != 'h2pc' and alg_name != 'mmhc'):
                    result = {}

                    result['Algorithm'] = alg_name
                    result['No. of calls'] = calls
                    result['Duration (s)'] = duration
                    rsmax2_time_results = rsmax2_time_results.append(result, ignore_index=True)

            # Read in the preprocessed results so we can tack on the time results
            rsmax2_existing_file = f"C:/Users/order/Documents/Oxford/4th Year/4YP/Pyro practice/data/{dims[0]}x{dims[1]}/R results/results_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/rsmax2_{dims[0]}x{dims[1]}_{num}_samples{i}.csv"
            rsmax2_existing_data = pd.read_csv(rsmax2_existing_file)
            rsmax2_existing_data = rsmax2_existing_data.drop(labels=[rsmax2_existing_data.columns[0]], axis=1)
            rsmax2_idx = rsmax2_existing_data.columns.get_loc('No. of calls')

            rsmax2_existing_data.insert(rsmax2_idx + 1, 'Duration (s)', rsmax2_time_results['Duration (s)'])


            data_file = f'{dims[0]}x{dims[1]}_{num}_samples{i}.csv'

            write_path = os.path.dirname(os.path.abspath(__file__)) + f'/data/{dims[0]}x{dims[1]}/R results/results_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/Version 2/'
            constraint_existing_data.to_csv(write_path + f'constraint_based_' + data_file, index=False)
            score_existing_data.to_csv(write_path + f'score_based_' + data_file, index=False)
            hybrid_existing_data.to_csv(write_path + f'hybrid_' + data_file, index=False)
            rsmax2_existing_data.to_csv(write_path + f'rsmax2_' + data_file, index=False)
            print('Completed ' + data_file)


# Plotting
# true_labels = cdt.metrics.retrieve_adjacency_matrix(ground_truth)
# print('Ground truth is \n', true_labels)
#
# fig1, ax1 = plt.subplots()
# ax1.set_title('IAMB Algorithm')
# fixed_nodes = position_nodes(variables)
# nx.draw(output, pos=fixed_nodes, with_labels=True)
# plt.show()
