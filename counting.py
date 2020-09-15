"""
An implementation of Giscard/Kriege/Wilson's (https://arxiv.org/pdf/1612.05531.pdf) algorithm
for counting simple cycles/paths up to a given length, based on an algebraic combinatorics formula.
"""
from typing import Set

import numpy as np
import scipy.special
import networkx as nx


def number_cycles(G: nx.DiGraph, cycle_len: np.ndarray):
    """
    Implementation of the generating function involution formula for gamma_ell.
    """
    sign = np.power(-1, cycle_len)
    combinatorial_sum = np.zeros(shape=cycle_len.shape)

    G_undir = G.to_undirected()
    for subset in connected_induced_undirected_subgraphs(G_undir, max_size=np.max(cycle_len)):
        H = nx.induced_subgraph(G, subset)
        A_H = nx.to_numpy_matrix(H)
        H_nodes = set(H.nodes())
        num_H_neighbors = len([
            v for v in range(len(G))
            if (v not in H) and has_edge(G_undir, H_nodes, v)
        ])
        combinatorial_sum += (
            scipy.special.binom(num_H_neighbors, cycle_len - len(H))
            * (+1 if len(H) % 2 == 0 else -1)
            * np.array([np.trace(np.linalg.matrix_power(A_H, _len)) for _len in cycle_len])
        )
    return (sign / cycle_len) * combinatorial_sum


def has_edge(G: nx.Graph, S: Set, v: int):
    """
    Outputs True if there is an edge between S and v, in undirected graph G.
    """
    return len(set(G[v]).intersection(S)) > 0


def connected_induced_undirected_subgraphs(G: nx.Graph, max_size: int) -> nx.Graph:
    """
    Enumerates over all connected induced undirected subgraphs by depth-first search.
    Implementation of [1996 Avis/Fukuda, "Reverse search for enumeration"].
    """
    n_vtx = len(G.nodes())

    # Note: S = {s} is the set of an empty set (s = EMPTY).
    v = set()
    f_v, adj_v = CIS_oracles(v, G, max_size)
    j = -1

    while len(v) > 0 or j < n_vtx-1:
        while j < n_vtx-1:
            j += 1
            next = adj_v(j)
            if next is not None:
                f_next, adj_next = CIS_oracles(next, G, max_size)
                if f_next == v:
                    yield next
                    v = next
                    f_v = f_next
                    adj_v = adj_next
                    j = -1
        if len(v) > 0:
            # Restore j.
            u = v
            v = f_v
            f_v, adj_v = CIS_oracles(v, G, max_size)
            j = 0
            while adj_v(j) != u:
                j = j + 1
                if j >= n_vtx:
                    break


def CIS_oracles(U: Set, G: nx.Graph, max_size: int):
    if len(U) == 0:
        f_cis = None
        adj_cis = lambda k: {k}
    elif len(U) == 1:
        v = U.__iter__().__next__()
        f_cis = set()

        def adj_cis(k: int):
            if k == v:
                return set()
            else:
                if k in G[v]:
                    return {v, k}
                else:
                    return None
    elif len(U) > 1:
        subgraph = G.subgraph(U)
        articulations = set(articulation_points(subgraph))
        f_cis = U.difference({
            min(set(subgraph.nodes).difference(articulations))
        })

        def adj_cis(k: int):
            if k in U:
                if k in articulations:
                    return None
                else:
                    return U.difference({k})
            else:
                if len(U) < max_size and len(set(G[k]).intersection(U)) > 0:
                    return U.union({k})
                else:
                    return None
    return f_cis, adj_cis


def articulation_points(G: nx.Graph):
    """
    Enumerate all "articulation points" (e.g. vertices whose removal splits up the graph into separate c.c's) of G.
    An implementation of Section 7.5 from https://doc.lagout.org/Alfred%20V.%20Aho%20-%20Data%20Structures%20and%20Algorithms.pdf.
    """
    if len(G.nodes()) <= 2:
        return
    root = G.nodes.__iter__().__next__()
    dfs_tree = DepthFirstSearchTree(G.nodes(), root)
    dfs_build(G, root, dfs_tree)

    lows = {u: float("inf") for u in G.nodes()}
    annotate_lows(dfs_tree, root, lows)
    for v in G:
        if v == root:
            if len(dfs_tree.neighbors(v)) > 1:
                yield v
        else:
            for w in dfs_tree.neighbors(v):
                if lows[w] >= dfs_tree.dfnumber(v):
                    yield v
                    break


class DepthFirstSearchTree(object):
    def __init__(self, vertices, root):
        self.nodes = dict()
        self.counter = 0
        self.adjacency = {v: [] for v in vertices}
        self.back_adjacency = {v: [] for v in vertices}
        self.add_node(root)

    def add_node(self, v):
        self.nodes[v] = self.counter
        self.counter += 1

    def add_child(self, parent, child):
        self.adjacency[parent].append(child)
        self.add_node(child)

    def add_back_edge(self, u, v):
        self.back_adjacency[u].append(v)
        self.back_adjacency[u].append(v)

    def contains(self, u):
        return u in self.nodes

    def neighbors(self, u):
        return self.adjacency[u]

    def back_neighbors(self, u):
        return self.back_adjacency[u]

    def dfnumber(self, u):
        return self.nodes[u]


def dfs_build(G, u, dfs_tree):
    for v in G[u]:
        if dfs_tree.contains(v):
            # v already visited
            dfs_tree.add_back_edge(u, v)
        else:
            # v not visited yet
            dfs_tree.add_child(u, v)
            dfs_build(G, v, dfs_tree)


def annotate_lows(dfs_tree, v, lows):
    children = dfs_tree.neighbors(v)
    if len(children) == 0:
        lows[v] = min(
            dfs_tree.dfnumber(v),
            min((dfs_tree.dfnumber(z) for z in dfs_tree.back_neighbors(v)), default=float("inf"))
        )
    else:
        for y in children:
            annotate_lows(dfs_tree, y, lows)
        lows[v] = min(
            dfs_tree.dfnumber(v),
            min((dfs_tree.dfnumber(z) for z in dfs_tree.back_neighbors(v)), default=float("inf")),
            min((lows[y] for y in children), default=float("inf"))
        )
