import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_edges_from([
    ("Network", "Router"),
    ("Network", "Computer"),
    ("Router", "Internet"),
    ("Computer", "User")
])

nx.draw(G, with_labels=True, node_color="lightblue",
        node_size=2500, arrows=True)
plt.show()
