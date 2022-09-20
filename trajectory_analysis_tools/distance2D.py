import networkx as nx
import numpy as np
import replay_trajectory_classification
from scipy.spatial.distance import cdist


def make_track_graph2D_from_environment(
        environment: replay_trajectory_classification.environments.Environment
) -> nx.Graph:
    """Creates a graph of the position where on track nodes are
    connected by edges.

    Parameters
    ----------
    environment : replay_trajectory_classification.environments.Environment

    Returns
    -------
    track_graph : nx.Graph
    """

    track_graph = nx.Graph()

    for node_id, node_position in enumerate(environment.place_bin_centers_):
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


def get_2D_distance(
    track_graph: nx.Graph,
    position1: np.ndarray,
    position2: np.ndarray
) -> np.ndarray:
    """Distance of two points along the graph of the track.


    Parameters
    ----------
    track_graph : nx.Graph
    position1 : np.ndarray, shape (n_time, 2)
    position2 : np.ndarray, shape (n_time, 2)

    Returns
    -------
    distance : np.ndarray
    """

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


def head_direction_simliarity(
        head_position: np.ndarray,
        head_direction: np.ndarray,
        position2: np.ndarray,
) -> np.ndarray:
    """Cosine similarity of the head direction vector with the vector from the
    animal's head to MAP estimate of the decoded position.

    Parameters
    ----------
    head_position : np.ndarray, shape (n_time, 2)
    head_direction : np.ndarray, shape (n_time, 2)
    position2 : np.ndarray, shape (n_time, 2)

    Returns
    -------
    cosine_similarity : np.ndarray, shape (n_time,)

    """

    head_direction = head_direction.squeeze()

    if head_position.ndim < 2:
        head_position = head_position[np.newaxis]
    if position2.ndim < 2:
        position2 = position2[np.newaxis]

    position2_direction = np.arctan2(
        position2[:, 1] - head_position[:, 1],
        position2[:, 0] - head_position[:, 0])

    return np.cos(head_direction - position2_direction)


def get_ahead_behind_distance2D(
        track_graph: nx.Graph,
        head_position: np.ndarray,
        head_direction: np.ndarray,
        map_position: np.ndarray,
) -> np.ndarray:
    """Distance of the MAP decoded position to the animal's head position where
     the sign indicates if the decoded position is in front of the
     head (positive) or behind (negative).

    Parameters
    ----------
    track_graph : nx.Graph
    head_position : np.ndarray, shape (n_time, 2)
    head_direction : np.ndarray, shape (n_time, 2)
    map_position : np.ndarray, shape (n_time, 2)

    Returns
    -------
    ahead_behind_distance : np.ndarray
    """

    distance = get_2D_distance(
        track_graph,
        head_position,
        map_position,
    )

    direction_similarity = head_direction_simliarity(
        head_position,
        head_direction,
        map_position,
    )
    ahead_behind = np.sign(direction_similarity)
    ahead_behind[np.isclose(ahead_behind, 0.0)] = 1.0

    ahead_behind_distance = ahead_behind * distance

    return ahead_behind_distance
