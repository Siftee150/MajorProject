import networkx as nx
from torch_geometric.utils.convert import from_networkx

nodes = ['1', '5', '28']
edges = [('1', '5'), ('5', '28')]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

x = [['0.7844669818878174', '-0.40328940749168396', '-0.9366764426231384'], ['0.14061762392520905',
                                                                             '-1.1449155807495117', '-0.1811756044626236'], ['-1.8840126991271973', '-1.2096494436264038', '1.0780194997787476']]

pyg_graph = from_networkx(G)
print(pyg_graph)
