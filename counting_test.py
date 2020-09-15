import numpy as np
from networkx import nx

G = nx.DiGraph()
G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 0)
G.add_edge(1, 3)
G.add_edge(0, 2)
print(number_cycles(G, np.array([3, 4])) == [2, 1])
