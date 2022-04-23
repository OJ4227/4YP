import pandas as pd
import numpy as np
import cdt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
import torch
from torch import tensor
from networkx import DiGraph
from matplotlib import pyplot as plt
import networkx as nx
from pgmpy.estimators import BDeuScore, BicScore
import os
from pgmpy.base import DAG
from pgmpy.metrics.metrics import structure_score


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


# num_of_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
num_of_samples = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

# list_of_dims = [[2, 2], [3, 3], [2, 4], [4, 2], [4, 4]]
list_of_dims = [[3, 3]]

for dims in list_of_dims:
    for num in num_of_samples:
        for i in range(3, 11):

            # Load in the file containing the data to calculate the information scores
            data_file = f'{dims[0]}x{dims[1]}_{num}_samples{i}.csv'
            read_path1 = os.path.dirname(
                os.path.abspath(__file__)) + f"/data/{dims[0]}x{dims[1]}/{dims[0]}x{dims[1]}_{num}_samples/" + data_file

            data = pd.read_csv(read_path1)

            variables = data.columns

            # Create the scoring objects
            bdeu = BDeuScore(data, equivalent_sample_size=1)
            bic = BicScore(data)

            read_path = os.path.dirname(os.path.abspath(
                __file__)) + f'/data/{dims[0]}x{dims[1]}/R outputs/outputs_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/'

            if num == 10000 or num == 20000:
                cd_files = os.listdir(read_path + 'CD/Times in seconds')
            else:
                cd_files = os.listdir(read_path + 'CD')


            adj_size = dims[0] * dims[1] * 2

            cd_results = pd.DataFrame()

            for file in cd_files:
                if num == 10000 or  num == 20000:
                    r_data = pd.read_csv(read_path + 'CD/Times in seconds/' + file)
                else:
                    r_data = pd.read_csv(read_path + 'CD/' + file)

                r_data = r_data.drop(labels=[r_data.columns[0]], axis=1)
                adj = r_data.iloc[:, :adj_size]
                adj = adj.to_numpy()
                alg_name = r_data.iloc[1, adj_size]
                time = r_data.iloc[1, adj_size + 1]
                alpha = r_data.iloc[1, adj_size + 2]
                # test = r_data.iloc[1, adj_size + 3]
                # p_value = r_data.iloc[1, adj_size + 4]
                # maximize = r_data.iloc[1, adj_size + 5]
                # score = r_data.iloc[1, adj_size + 6]

                output = nx.from_numpy_array(adj, create_using=nx.DiGraph)
                mapping = {i: variables[i] for i in range(output.number_of_nodes())}
                output = nx.relabel_nodes(output, mapping)

                ground_truth = create_ground_truth_adj(dims, variables)

                result = {}

                result['Algorithm'] = alg_name
                result['Alpha'] = alpha
                result['Duration (s)'] = time
                result['SHD'] = cdt.metrics.SHD(ground_truth, output)
                result['SID'] = cdt.metrics.SID(ground_truth, output)
                result['PR AUC'] = cdt.metrics.precision_recall(ground_truth, output)[0]
                result['ROC AUC'] = calculate_roc_auc(output, ground_truth)
                result['F1'] = calculate_f1(output, ground_truth)

                ## Check why F1 score is 0
                # true_labels = cdt.metrics.retrieve_adjacency_matrix(ground_truth)
                # pred = cdt.metrics.retrieve_adjacency_matrix(output, order_nodes=ground_truth.nodes() if isinstance(ground_truth,nx.DiGraph) else None)
                #
                # f1 = f1_score(true_labels.ravel(), pred.ravel())
                # precision = precision_score(true_labels.ravel(), pred.ravel())
                # recall = recall_score(true_labels.ravel(), pred.ravel())
                #
                # fixed_nodes = position_nodes(variables)
                # nx.draw(output, pos=fixed_nodes, with_labels=True)

                result['Accuracy'] = calculate_accuracy(output, ground_truth)
                result['BDeu'] = bdeu.score(output)
                result['BIC'] = bic.score(output)
                cd_results = cd_results.append(result, ignore_index=True)


            write_path = os.path.dirname(os.path.abspath(__file__)) + f'/data/{dims[0]}x{dims[1]}/R results/results_{dims[0]}x{dims[1]}_{num}_samples/{dims[0]}x{dims[1]}_{num}_samples{i}/'
            # write_path = os.path.dirname(os.path.abspath(__file__)) + f'/data/'
            # constraint_results.to_csv(write_path + f'constraint_based_' + data_file)
            # score_results.to_csv(write_path + f'score_based_' + data_file)
            # hybrid_results.to_csv(write_path + f'hybrid_' + data_file)
            # rsmax2_results.to_csv(write_path + f'rsmax2_' + data_file)
            cd_results.to_csv(write_path + f'cd_' + data_file)
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
