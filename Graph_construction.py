import networkx as nx
import pandas as pd
import numpy as np
from torch_geometric.data import Data


def construct_attribute_graph(data):
    # Create an empty attribute graph
    G = nx.Graph()

    # Get the total number of rows in the data
    total_rows = len(data)

    for index, row in data.iterrows():
        flow_id = index  # Use the index as the unique identifier for nodes
        features = row.drop('Label')  # Get all features except the 'Label' column
        label = row['Label']  # Get the label

        # Add nodes and attributes
        G.add_node(flow_id, features=features.to_dict(), label=label)

    # Create edges
    num_nodes = len(G.nodes)
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            u = list(G.nodes)[i]
            v = list(G.nodes)[j]
            if u != v:  # Connect only different nodes
                u_features = G.nodes[u]['features']
                v_features = G.nodes[v]['features']
                # Count the number of common key-value pairs
                common_features = sum(u_features.get(key) == v_features.get(key) for key in u_features)
                if common_features > 0:  # If there are common key-value pairs
                    G.add_edge(u, v, weight=common_features)  # Add weight

    # Compute weighted clustering coefficients and betweenness centrality
    weighted_clustering_coefficients = nx.clustering(G, weight='weight')
    betweenness_centrality = nx.betweenness_centrality(G)

    # Add clustering coefficients and betweenness centrality to node features
    for node in G.nodes:
        G.nodes[node]['features']['weighted_clustering_coefficient'] = weighted_clustering_coefficients[node]
        G.nodes[node]['features']['betweenness_centrality'] = betweenness_centrality[node]

    adjacency_matrix = nx.adjacency_matrix(G)
    feature_matrix = pd.DataFrame.from_dict(dict(G.nodes(data='features')), orient='index')

    # Convert adjacency matrix and feature matrix to PyTorch Geometric Data object
    edge_index = np.array(adjacency_matrix.nonzero())
    attribute_graph = Data(x=feature_matrix, edge_index=edge_index, y=label)

    return attribute_graph