import matplotlib.pyplot as plt
import networkx as nx

def exportGraph(G, filename):
    pos = nx.planar_layout(G, scale=2)
    # pos = nx.spring_layout(G, scale=2)
    # pos = nx.spiral_layout(G, scale=2)
    # nodes
    # nx.draw_networkx_nodes(G, pos, node_size=1000)

    # node labels
    # nx.draw_networkx_labels(G, pos, font_size=8, font_family="Times New Roman")

    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw(G, pos, with_labels=True, node_size=1000, font_size=8, font_family="Times New Roman", arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, font_family="Times New Roman", font_color='red')

    ax = plt.gca()
    ax.margins(0.1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)