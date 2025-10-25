import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import networkx as nx

G = nx.Graph()
G.add_node("M1", layer=0)
G.add_node("M2", layer=0)
G.add_node("M3", layer=0)
G.add_node("M4", layer=0)
G.add_edge("M1", "S2")
G.add_edge("M1", "S3")
G.add_edge("M1", "S5")
G.add_edge("M2", "S1")
G.add_edge("M2", "S4")
G.add_edge("M2", "S5")
G.add_edge("M2", "S6")
G.add_edge("M3", "S6")
G.add_edge("M3", "S7")
G.add_edge("M4", "S7")
G.add_edge("M4", "S8")
G.add_node("S1", layer=1)
G.add_node("S2", layer=1)
G.add_node("S3", layer=1)
G.add_node("S4", layer=1)
G.add_node("S5", layer=1)
G.add_node("S6", layer=1)
G.add_node("S7", layer=1)
G.add_node("S8", layer=1)
G.add_edge("S1", "Z1")
G.add_edge("S1", "Z2")
G.add_edge("S2", "Z2")
G.add_edge("S2", "Z3")
G.add_edge("S3", "Z2")
G.add_edge("S3", "Z3")
G.add_edge("S4", "Z1")
G.add_edge("S5", "Z4")
G.add_edge("S6", "Z4")
G.add_edge("S7", "Z4")
G.add_edge("S8", "Z4")
G.add_node("Z1", layer=2)
G.add_node("Z2", layer=2)
G.add_node("Z3", layer=2)
G.add_node("Z4", layer=2)

h = 0.5
pos = {
    "M1":(2, 0), "M2":(3.5, 0), "M3":(6, 0), "M4":(7, 0),
    "S1":(0.5, h), "S2":(1.5, h), "S3":(2.5, h), "S4":(3.5, h), "S5":(4.5, h), "S6":(5.5, h), "S7":(6.5, h), "S8":(7.5, h),
    "Z1":(1, 2*h), "Z2":(2, 2*h), "Z3":(3, 2*h), "Z4":(6, 2*h),
}

colors = ["#66b3e7", "#fde397", "#d9958f"]
for node, layer in G.nodes(data='layer'):
    if layer is not None:
        G.nodes[node]['color'] = colors[layer]

nx.draw_networkx_edges(G, pos, edge_color="#568ec2", width=2)
nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[node]['color'] for node in G.nodes], node_size=1200)
nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_color='black', 
                        font_size=15, font_weight='bold')

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              label='minimum' if i == 2 else f'index-{2-i}', 
                              markerfacecolor=color, markersize=10) 
                   for i, color in enumerate(colors)]
plt.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=10)

plt.xlim(min(pos.values(), key=lambda x: x[0])[0]-1, max(pos.values(), key=lambda x: x[0])[0]+1)
plt.ylim(min(pos.values(), key=lambda x: x[1])[1]-0.5, max(pos.values(), key=lambda x: x[1])[1]+0.5)

plt.gca().invert_yaxis()
plt.axis('off')
# lt.savefig("Tree2.png", dpi=300)
plt.show()