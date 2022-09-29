import networkx as nx
import numpy as np
import replay_trajectory_classification
from scipy.spatial.distance import cdist


def make_track_graph2D_from_environment(
    environment: replay_trajectory_classification.environments.Environment,
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
        x_inds, y_inds = np.meshgrid(
            x_ind + np.asarray([-1, 0, 1]), y_ind + np.asarray([-1, 0, 1])
        )
        adj_edges = environment.is_track_interior_[x_inds, y_inds]
        adj_edges[1, 1] = False

        node_id = np.ravel_multi_index(
            (x_ind, y_ind), environment.centers_shape_, order="F"
        )
        adj_node_ids = np.ravel_multi_index(
            (x_inds[adj_edges], y_inds[adj_edges]),
            environment.centers_shape_,
            order="F",
        )
        edges.append(np.concatenate(np.meshgrid(node_id, adj_node_ids), axis=1))

    edges = np.concatenate(edges)

    for (node1, node2) in edges:
        pos1 = np.asarray(track_graph.nodes[node1]["pos"])
        pos2 = np.asarray(track_graph.nodes[node2]["pos"])
        distance = np.linalg.norm(pos1 - pos2)
        track_graph.add_edge(node1, node2, distance=distance)

    for edge_id, edge in enumerate(track_graph.edges):
        track_graph.edges[edge]["edge_id"] = edge_id

    return track_graph


def find_closest_node(pos, node_positions):
    return np.argmin(cdist(node_positions, pos[np.newaxis]))


def get_map_estimate_direction_from_track_graph(
    track_graph, head_position, map_estimate
):
    node_positions = nx.get_node_attributes(track_graph, "pos")
    node_positions = np.asarray(list(node_positions.values()))

    map_estimate_direction = []

    for head_pos, map_est in zip(head_position, map_estimate):
        head_position_node = find_closest_node(head_pos)
        map_estimate_node = find_closest_node(map_est)

        first_node_on_path = nx.shortest_path(
            track_graph,
            source=head_position_node,
            target=map_estimate_node,
            weight="distance",
        )[1]

        head_position_node_pos = track_graph.nodes[head_position_node]["pos"]
        first_node_on_path_pos = track_graph.nodes[first_node_on_path]["pos"]

        map_estimate_direction.append(
            np.arctan2(
                first_node_on_path_pos[1] - head_position_node_pos[1],
                first_node_on_path_pos[0] - head_position_node_pos[0],
            )
        )

    return np.asarray(map_estimate_direction)


def get_2D_distance(
    position1: np.ndarray, position2: np.ndarray, track_graph: nx.Graph = None,
) -> np.ndarray:
    """Distance of two points along the graph of the track.


    Parameters
    ----------
    position1 : np.ndarray, shape (n_time, 2)
    position2 : np.ndarray, shape (n_time, 2)
    track_graph : nx.Graph or None

    Returns
    -------
    distance : np.ndarray

    """

    if position1.ndim < 2:
        position1 = position1[np.newaxis]
    if position2.ndim < 2:
        position2 = position2[np.newaxis]

    if track_graph is None:
        distance = np.linalg.norm(position1 - position2, axis=1)
    else:
        node_positions = nx.get_node_attributes(track_graph, "pos")
        node_positions = np.asarray(list(node_positions.values()))

        distance = list()

        for pos1, pos2 in zip(position1, position2):
            node_id1 = find_closest_node(pos1, node_positions)
            node_id2 = find_closest_node(pos2, node_positions)
            distance.append(
                nx.shortest_path_length(
                    track_graph, source=node_id1, target=node_id2, weight="distance"
                )
            )
        distance = np.asarray(distance)

    return np.asarray(distance)


def head_direction_simliarity(
    head_position: np.ndarray,
    head_direction: np.ndarray,
    map_estimate: np.ndarray,
    track_graph: nx.Graph = None,
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
    if map_estimate.ndim < 2:
        map_estimate = map_estimate[np.newaxis]

    if track_graph is None:
        map_estimate_direction = np.arctan2(
            map_estimate[:, 1] - head_position[:, 1],
            map_estimate[:, 0] - head_position[:, 0],
        )
    else:
        map_estimate_direction = get_map_estimate_direction_from_track_graph(
            track_graph, head_position, map_estimate
        )

    return np.cos(head_direction - map_estimate_direction)


def get_ahead_behind_distance2D(
    head_position: np.ndarray,
    head_direction: np.ndarray,
    map_position: np.ndarray,
    track_graph: nx.Graph = None,
) -> np.ndarray:
    """Distance of the MAP decoded position to the animal's head position where
     the sign indicates if the decoded position is in front of the
     head (positive) or behind (negative).

    Parameters
    ----------
    head_position : np.ndarray, shape (n_time, 2)
    head_direction : np.ndarray, shape (n_time, 2)
    map_position : np.ndarray, shape (n_time, 2)
    track_graph : nx.Graph or None

    Returns
    -------
    ahead_behind_distance : np.ndarray
    """

    distance = get_2D_distance(head_position, map_position, track_graph)

    direction_similarity = head_direction_simliarity(
        head_position, head_direction, map_position, track_graph,
    )
    ahead_behind = np.sign(direction_similarity)
    ahead_behind[np.isclose(ahead_behind, 0.0)] = 1.0

    ahead_behind_distance = ahead_behind * distance

    return ahead_behind_distance
