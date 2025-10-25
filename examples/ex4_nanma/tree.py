import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import networkx as nx

G = nx.Graph()
G.add_node("M1", layer=0)
G.add_node("M2", layer=0)
G.add_node("M3", layer=0)
G.add_edge("M1", "S1")
G.add_edge("M1", "S2")
G.add_edge("M2", "S1")
G.add_edge("M2", "S2")
G.add_edge("M2", "S3")
G.add_edge("M2", "S4")
G.add_edge("M2", "S6")
G.add_edge("M3", "S3")
G.add_edge("M3", "S4")
G.add_edge("M3", "S5")
G.add_edge("M3", "S6")
G.add_node("S1", layer=1)
G.add_node("S2", layer=1)
G.add_node("S3", layer=1)
G.add_node("S4", layer=1)
G.add_node("S5", layer=1)
G.add_node("S6", layer=1)
G.add_edge("S1", "Z1")
G.add_edge("S1", "Z2")
G.add_edge("S2", "Z1")
G.add_edge("S2", "Z2")
G.add_edge("S3", "Z1")
G.add_edge("S4", "Z2")
G.add_edge("S4", "Z3")
G.add_edge("S5", "Z1")
G.add_edge("S5", "Z3")
G.add_edge("S6", "Z2")
G.add_edge("S6", "Z3")
G.add_node("Z1", layer=2)
G.add_node("Z2", layer=2)
G.add_node("Z3", layer=2)

h = 0.5
pos = {
    "M1":(1, 0), "M2":(2.5, 0), "M3":(4, 0),
    "S1":(0.5, h), "S2":(1.5, h), "S3":(2.5, h), "S4":(3.5, h), "S5":(4.5, h), "S6":(5.5, h),
    "Z1":(1.5, 2*h), "Z2":(3, 2*h), "Z3":(5, 2*h),
}

colors = ["#66b3e7", "#fde397", "#d9958f"]
for node, layer in G.nodes(data='layer'):
    if layer is not None:
        G.nodes[node]['color'] = colors[layer]

nx.draw_networkx_edges(G, pos, edge_color="#568ec2", width=2)
nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[node]['color'] for node in G.nodes], node_size=1600)
nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_color='black', 
                        font_size=20, font_weight='bold')

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              label='minimum' if i == 2 else f'index-{2-i}', 
                              markerfacecolor=color, markersize=10) 
                   for i, color in enumerate(colors)]
plt.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=12)

plt.xlim(min(pos.values(), key=lambda x: x[0])[0]-0.5, max(pos.values(), key=lambda x: x[0])[0]+0.5)
plt.ylim(min(pos.values(), key=lambda x: x[1])[1]-0.5, max(pos.values(), key=lambda x: x[1])[1]+0.5)

plt.gca().invert_yaxis()
plt.axis('off')
# plt.savefig("Tree0.png", dpi=300)
plt.show()