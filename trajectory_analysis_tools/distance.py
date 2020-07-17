import networkx as nx
import numpy as np
from loren_frank_data_processing.track_segment_classification import (
    get_track_segments_from_graph, project_points_to_segment)


def _get_MAP_estimate_2d_position_edges(posterior, track_graph, decoder):
    # Get 2D position on track from decoder MAP estimate
    map_position_ind = (
        posterior.where(decoder.is_track_interior_).argmax(
            "position", skipna=True).values
    )
    map_position_2d = decoder.place_bin_center_2D_position_[
        map_position_ind]

    # Figure out which track segment it belongs to
    track_segment_id = decoder.place_bin_center_ind_to_edge_id_[
        map_position_ind]
    map_edges = np.array(track_graph.edges)[track_segment_id]

    return map_position_2d, map_edges


def _get_animal_2d_projected_position_edges(
        track_graph, position_2D, track_segment_id):
    # Get animal's 2D position projected onto track
    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_positions = project_points_to_segment(
        track_segments, position_2D)
    n_time = projected_track_positions.shape[0]
    actual_projected_position = projected_track_positions[(
        np.arange(n_time), track_segment_id)]

    # Add animal's position at time to track graph
    actual_edges = np.array(track_graph.edges)[track_segment_id]

    return actual_projected_position, actual_edges


def add_node(pos, edge, graph, node_name):
    node1, node2 = edge
    x3, y3 = pos

    x1, y1 = graph.nodes[node1]['pos']
    left_distance = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    nx.add_path(graph, [node1, node_name], distance=left_distance)

    x2, y2 = graph.nodes[node2]['pos']
    right_distance = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    nx.add_path(
        graph, [node_name, node2], distance=right_distance)


def calculate_replay_distance(
        posterior, track_graph, decoder, position_2D, track_segment_id):
    track_segment_id = np.asarray(track_segment_id).astype(int).squeeze()
    position_2D = np.asarray(position_2D)

    map_position_2d, map_edges = _get_MAP_estimate_2d_position_edges(
        posterior, track_graph, decoder)
    (actual_projected_position,
     actual_edges) = _get_animal_2d_projected_position_edges(
        track_graph, position_2D, track_segment_id)

    copy_graph = track_graph.copy()
    replay_distance_from_animal_position = []

    for actual_pos, actual_edge, map_pos, map_edge in zip(
            actual_projected_position, actual_edges, map_position_2d,
            map_edges):

        # Add actual position node
        add_node(actual_pos, actual_edge, copy_graph, 'actual_position')
        add_node(map_pos, map_edge, copy_graph, 'map_position')
        if np.all(actual_edge == map_edge):
            (x1, y1), (x2, y2) = actual_pos, map_pos
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            nx.add_path(
                copy_graph, ['actual_position', 'map_position'],
                distance=distance)
        replay_distance_from_animal_position.append(
            nx.shortest_path_length(copy_graph, source='actual_position',
                                    target='map_position', weight='distance'))
        copy_graph.remove_node('actual_position')
        copy_graph.remove_node('map_position')

    return np.asarray(replay_distance_from_animal_position)


def points_toward_node(track_graph, edge, head_direction):
    """Given an edge, determine the node the head is pointed toward

    Parameters
    ----------
    track_graph : networkx.Graph
    edge : array-like, shape (2,)
    head_direction : array-like
        Angle of head in radians

    Returns
    -------
    node : object

    """
    edge = np.asarray(edge)
    node1_pos = np.asarray(track_graph.nodes[edge[0]]["pos"])
    node2_pos = np.asarray(track_graph.nodes[edge[1]]["pos"])
    edge_vector = node2_pos - node1_pos
    head_vector = np.asarray([np.cos(head_direction), np.sin(head_direction)])

    return edge[(edge_vector @ head_vector >= 0).astype(int)]


def get_distance_between_nodes(track_graph, node1, node2):
    node1_pos = np.asarray(track_graph.nodes[node1]["pos"])
    node2_pos = np.asarray(track_graph.nodes[node2]["pos"])
    return np.sqrt(np.sum((node1_pos - node2_pos) ** 2))


def get_ahead_or_behind(
    track_graph, actual_pos, actual_edge, head_direction, mental_pos,
    mental_edge
):
    """

    Parameters
    ----------
    track_graph : nx.Graph
    actual_pos : array-like, shape (2,)
    actual_edge : array-like, shape (2,)
    head_direction : float
    mental_pos : array-like, shape (2,)
    mental_edge : array-like, shape (2,)

    Returns
    -------
    ahead_behind : {-1, 0, 1}
        -1 is behind, 1 is ahead, 0 is same

    """
    if np.allclose(actual_pos, mental_pos):
        return 0
    else:
        track_graph.add_node("actual_position", pos=actual_pos)
        track_graph.add_node("head", pos=actual_pos)
        track_graph.add_node("mental_position", pos=mental_pos)

        # determine which node head is pointing towards
        node_ahead = points_toward_node(
            track_graph, actual_edge, head_direction)
        node_behind = actual_edge[~np.isin(actual_edge, node_ahead)][0]

        # insert edges between nodes
        if np.all(actual_edge == mental_edge):  # if all on same edge
            same_side = (
                get_distance_between_nodes(
                    track_graph, "actual_position", "mental_position") <=
                get_distance_between_nodes(
                    track_graph, "actual_position", node_ahead))
            if same_side:
                node_order = [
                    node_ahead,
                    "mental_position",
                    "head",
                    "actual_position",
                    node_behind,
                ]
            else:
                node_order = [
                    node_ahead,
                    "head",
                    "actual_position",
                    "mental_position",
                    node_behind,
                ]
        else:
            node_order = [node_ahead, "head", "actual_position", node_behind]

            distance = get_distance_between_nodes(
                track_graph, mental_edge[0], "mental_position")
            track_graph.add_edge(
                mental_edge[0], "mental_position", distance=distance)

            distance = get_distance_between_nodes(
                track_graph, "mental_position", mental_edge[1])
            track_graph.add_edge(
                "mental_position", mental_edge[1], distance=distance)

        for node1, node2 in zip(node_order[:-1], node_order[1:]):
            distance = get_distance_between_nodes(track_graph, node1, node2)
            track_graph.add_edge(node1, node2, distance=distance)

        # Find shortest path in terms of nodes
        path = nx.shortest_path(
            track_graph,
            source="actual_position",
            target="mental_position",
            weight="distance",
        )

        # Cleanup: remove inserted nodes
        track_graph.remove_node("actual_position")
        track_graph.remove_node("head")
        track_graph.remove_node("mental_position")

        return 1 if "head" in path else -1


def calculate_ahead_behind(posterior, track_graph, decoder, position_info):

    map_position_2d, map_edges = _get_MAP_estimate_2d_position_edges(
        posterior, track_graph, decoder)
    actual_projected_position = position_info[
        ["projected_x_position", "projected_y_position"]
    ].values
    track_segment_id = position_info.track_segment_id.values.astype(
        int).squeeze()
    actual_edges = np.asarray(track_graph.edges)[track_segment_id]
    head_directions = position_info.head_direction.values

    copy_graph = track_graph.copy()
    ahead_behind = []

    for actual_pos, actual_edge, head_dir, map_pos, map_edge in zip(
            actual_projected_position, actual_edges, head_directions,
            map_position_2d, map_edges):
        ahead_behind.append(get_ahead_or_behind(
            copy_graph, actual_pos, actual_edge, head_dir, map_pos,
            map_edge
        ))

    return np.asarray(ahead_behind)
