import networkx as nx
import numpy as np
import replay_trajectory_classification
from scipy.spatial.distance import cdist


def make_track_graph2D_from_environment(
        environment: replay_trajectory_classification.environments.Environment
) -> nx.Graph:

    track_graph = nx.Graph()
    is_track_interior = environment.is_track_interior_.ravel(order='F')
    for node_id, node_position in zip(
            np.nonzero(is_track_interior)[0],
            environment.place_bin_centers_[is_track_interior]):
        track_graph.add_node(node_id, pos=tuple(node_position))

    edges = []
    for x_ind, y_ind in zip(*np.nonzero(environment.is_track_interior_)):
        x_inds, y_inds = np.meshgrid(x_ind + np.asarray([-1, 0, 1]),
                                     y_ind + np.asarray([-1, 0, 1]))
        adj_edges = environment.is_track_interior_[x_inds, y_inds]
        adj_edges[1, 1] = False

        node_id = np.ravel_multi_index(
            (x_ind, y_ind),
            environment.centers_shape_,
            order='F')
        adj_node_ids = np.ravel_multi_index(
            (x_inds[adj_edges], y_inds[adj_edges]),
            environment.centers_shape_,
            order='F')
        edges.append(np.concatenate(
            np.meshgrid(node_id, adj_node_ids), axis=1))

    edges = np.concatenate(edges)

    for (node1, node2) in edges:
        pos1 = np.asarray(track_graph.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        track_graph.add_edge(node1, node2, distance=distance)

    for edge_id, edge in enumerate(track_graph.edges):
        track_graph.edges[edge]["edge_id"] = edge_id

    return track_graph


def get_distance_on_graph(
    track_graph: nx.Graph,
    position1: np.ndarray,
    position2: np.ndarray
) -> np.ndarray:

    if position1.ndim < 2:
        position1 = position1[np.newaxis]
    if position2.ndim < 2:
        position2 = position2[np.newaxis]

    node_positions = nx.get_node_attributes(track_graph, 'pos')
    node_positions = np.asarray(list(node_positions.values()))

    distance = list()

    for pos1, pos2 in zip(position1, position2):
        node_id1 = np.argmin(cdist(node_positions, pos1[np.newaxis]))
        node_id2 = np.argmin(cdist(node_positions, pos2[np.newaxis]))
        distance.append(nx.shortest_path_length(
            track_graph, source=node_id1, target=node_id2, weight='distance'))

    return np.asarray(distance)
