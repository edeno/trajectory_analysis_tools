import networkx as nx
import numpy as np
import pandas as pd


def _get_MAP_estimate_2d_position_edges(posterior, track_graph, decoder):
    # Get 2D position on track from decoder MAP estimate
    map_position_ind = (
        posterior.where(decoder.is_track_interior_).argmax(
            "position", skipna=True).values
    )
    mental_position_2d = decoder.place_bin_center_2D_position_[
        map_position_ind]

    # Figure out which track segment it belongs to
    track_segment_id = decoder.place_bin_center_ind_to_edge_id_[
        map_position_ind]
    mental_position_edges = np.array(track_graph.edges)[track_segment_id]

    return mental_position_2d, mental_position_edges


def _points_toward_node(track_graph, edge, head_direction):
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


def _get_distance_between_nodes(track_graph, node1, node2):
    node1_pos = np.asarray(track_graph.nodes[node1]["pos"])
    node2_pos = np.asarray(track_graph.nodes[node2]["pos"])
    return np.sqrt(np.sum((node1_pos - node2_pos) ** 2))


def _setup_track_graph(track_graph, actual_pos, actual_edge, head_direction,
                       mental_pos, mental_edge):
    """Takes the track graph and add nodes for the animal's actual position,
    mental position, and head direction.

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
    track_graph

    """
    track_graph.add_node("actual_position", pos=actual_pos)
    track_graph.add_node("head", pos=actual_pos)
    track_graph.add_node("mental_position", pos=mental_pos)

    # determine which node head is pointing towards
    node_ahead = _points_toward_node(
        track_graph, actual_edge, head_direction)
    node_behind = actual_edge[~np.isin(actual_edge, node_ahead)][0]

    # insert edges between nodes
    if np.all(actual_edge == mental_edge):  # actual and mental on same edge
        actual_pos_distance = _get_distance_between_nodes(
            track_graph, "actual_position", node_ahead)
        mental_pos_distance = _get_distance_between_nodes(
            track_graph, "mental_position", node_ahead)

        if actual_pos_distance < mental_pos_distance:
            node_order = [
                node_ahead,
                "head",
                "actual_position",
                "mental_position",
                node_behind
            ]
        else:
            node_order = [
                node_ahead,
                "mental_position",
                "head",
                "actual_position",
                node_behind
            ]
    else:  # actual and mental are on different edges
        node_order = [
            node_ahead,
            "head",
            "actual_position",
            node_behind
        ]

        distance = _get_distance_between_nodes(
            track_graph, mental_edge[0], "mental_position")
        track_graph.add_edge(
            mental_edge[0], "mental_position", distance=distance)

        distance = _get_distance_between_nodes(
            track_graph, "mental_position", mental_edge[1])
        track_graph.add_edge(
            "mental_position", mental_edge[1], distance=distance)

    for node1, node2 in zip(node_order[:-1], node_order[1:]):
        distance = _get_distance_between_nodes(track_graph, node1, node2)
        track_graph.add_edge(node1, node2, distance=distance)

    return track_graph


def _calculate_ahead_behind(track_graph, source="actual_position",
                            target="mental_position"):
    path = nx.shortest_path(
        track_graph,
        source=source,
        target=target,
        weight="distance",
    )

    return 1 if "head" in path else -1


def _calculate_distance(track_graph, source="actual_position",
                        target="mental_position"):
    return nx.shortest_path_length(track_graph, source=source,
                                   target=target, weight='distance')


def get_trajectory_data(posterior, track_graph, decoder, position_info,
                        direction_variable="head_direction"):
    (mental_position_2d,
     mental_position_edges) = _get_MAP_estimate_2d_position_edges(
        posterior, track_graph, decoder)
    actual_projected_position = np.asarray(position_info[
        ["projected_x_position", "projected_y_position"]
    ])
    track_segment_id = np.asarray(position_info.track_segment_id).astype(
        int).squeeze()
    actual_edges = np.asarray(track_graph.edges)[track_segment_id]
    directions = np.asarray(position_info[direction_variable])

    return (actual_projected_position, actual_edges, directions,
            mental_position_2d, mental_position_edges)


def get_distance_metrics(track_graph, actual_projected_position, actual_edges,
                         orientations, mental_position_2d,
                         mental_position_edges):

    copy_graph = track_graph.copy()
    mental_position_ahead_behind_animal = []
    mental_position_distance_from_animal = []

    for actual_pos, actual_edge, orientation, map_pos, map_edge in zip(
            actual_projected_position, actual_edges, orientations,
            mental_position_2d, mental_position_edges):
        # Insert nodes for actual position, mental position, head
        copy_graph = _setup_track_graph(
            copy_graph, actual_pos, actual_edge, orientation, map_pos,
            map_edge)

        # Get metrics
        mental_position_distance_from_animal.append(
            _calculate_distance(copy_graph, source="actual_position",
                                target="mental_position")
        )
        mental_position_ahead_behind_animal.append(
            _calculate_ahead_behind(copy_graph, source="actual_position",
                                    target="mental_position"))

        # Cleanup: remove inserted nodes
        copy_graph.remove_node("actual_position")
        copy_graph.remove_node("head")
        copy_graph.remove_node("mental_position")

    return pd.DataFrame(
        {
            "mental_position_ahead_behind_animal": mental_position_ahead_behind_animal,  # noqa
            "mental_position_distance_from_animal": mental_position_distance_from_animal,  # noqa
        })
