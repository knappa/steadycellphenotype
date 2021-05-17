from typing import Dict, List

import networkx as nx
import numpy as np


def connected_component_layout(g: nx.DiGraph):
    """
    lay out a graph with a single connected component,
    returns dictionary of positions and width/height of bounding box
    """

    # get attractor (fixed point or cycle)
    attractor_set = next(nx.attracting_components(g))
    cycle_len = len(attractor_set)

    # no guarantee the attractor set is in the proper order:
    base_point = next(iter(attractor_set))
    cycle = [base_point]
    # in python 3.8+ you have assignment expressions:
    # while (next_point := list(g.successors(cycle[-1]))[0]) != base_point:
    #    cycle.append(next_point)
    next_point = list(g.successors(cycle[-1]))[0]
    while next_point != base_point:
        cycle.append(next_point)
        next_point = list(g.successors(cycle[-1]))[0]

    pos = dict()

    visited_set = set()

    def get_num_leaves(parent):
        if parent in visited_set:
            return 0
        else:
            visited_set.add(parent)
        predecessors = [predecessor
                        for predecessor in g.predecessors(parent)
                        if predecessor != parent]
        if len(predecessors) == 0:
            return 1
        else:
            return sum(get_num_leaves(predecessor) for predecessor in predecessors)

    def recurse_layout(successor, radius: float, max_theta: float, min_theta: float):
        predecessors = [predecessor
                        for predecessor in g.predecessors(successor)
                        if predecessor != successor and predecessor not in pos]
        if len(predecessors) == 0:
            return

        angles = np.cumsum(np.array([0.0] + [get_num_leaves(predecessor) for predecessor in predecessors],
                                    dtype=np.float64))
        angles *= (max_theta - min_theta) / angles[-1] if angles[-1] != 0 else (max_theta - min_theta)
        angles += min_theta
        for n, predecessor in enumerate(predecessors):
            theta_n = (angles[n + 1] + angles[n]) / 2.0
            pos[predecessor] = radius * np.array([np.cos(theta_n),
                                                  np.sin(theta_n)])
            recurse_layout(predecessor, radius + 20, angles[n + 1], angles[n])

    # lay out the cycle:
    if cycle_len == 1:
        pos[base_point] = np.array([0.0, 0.0])
        recurse_layout(base_point, 20, 2 * np.pi, 0)
    else:
        angles = np.cumsum(np.array([0] + [get_num_leaves(point) for point in cycle],
                                    dtype=np.float64))
        angles *= 2 * np.pi / angles[-1] if angles[-1] != 0 else 2 * np.pi
        for n, point in enumerate(cycle):
            theta = (angles[n + 1] + angles[n]) / 2.0
            pos[point] = 20 * np.array([np.cos(theta), np.sin(theta)])
            recurse_layout(point, 20, angles[n + 1], angles[n])

    # move corner
    pos_array = np.array(list(pos.values()))
    offset = np.min(pos_array, axis=0)
    pos = {node: pt - offset for node, pt in pos.items()}
    return pos, np.max(pos_array, axis=0) - offset


def graph_layout(edge_lists: List[List[Dict[str, int]]], height_px, width_px):
    # lay out connected components, in bounding boxes. then offset
    # noinspection PyTypeChecker

    g: nx.DiGraph = nx.DiGraph()
    for edge_list in edge_lists:
        for edge in edge_list:
            g.add_edge(edge['source'], edge['target'])

    # noinspection PyTypeChecker
    components_layouts = [
        connected_component_layout(nx.subgraph_view(g, filter_node=lambda vertex: vertex in component_vertices))
        for component_vertices in nx.weakly_connected_components(g)]

    positions = dict()
    corner = np.array([-float(height_px) / 2, -float(width_px) / 2])
    running_y = 0.0
    for component_pos, geom in components_layouts:
        running_y = max(running_y, geom[1])
        for node in component_pos:
            positions[node] = component_pos[node] + corner
        corner += np.array([geom[0] + 1.0, 0])
        if corner[0] > 20.0:
            corner[0] = 0
            corner[1] += running_y
            running_y = 0.0
    return positions
