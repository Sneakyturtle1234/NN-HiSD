import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import networkx as nx

G = nx.Graph()
G.add_node("M1", layer=0)
G.add_node("M2", layer=0)
G.add_edge("M1", "S1")
G.add_edge("M1", "S2")
G.add_edge("M1", "S3")
G.add_edge("M2", "S4")
G.add_edge("M2", "S5")
G.add_edge("M2", "S6")
G.add_edge("M2", "S7")
G.add_node("S1", layer=1)
G.add_node("S2", layer=1)
G.add_node("S3", layer=1)
G.add_node("S4", layer=1)
G.add_node("S5", layer=1)
G.add_node("S6", layer=1)
G.add_node("S7", layer=1)
G.add_edge("S1", "Z1")
G.add_edge("S2", "Z1")
G.add_edge("S2", "Z2")
G.add_edge("S2", "Z3")
G.add_edge("S3", "Z2")
G.add_edge("S3", "Z4")
G.add_edge("S4", "Z3")
G.add_edge("S5", "Z3")
G.add_edge("S5", "Z4")
G.add_edge("S6", "Z4")
G.add_edge("S6", "Z5")
G.add_edge("S7", "Z5")
G.add_node("Z1", layer=2)
G.add_node("Z2", layer=2)
G.add_node("Z3", layer=2)
G.add_node("Z4", layer=2)
G.add_node("Z5", layer=2)

h = 0.5
pos = {
    "M1":(1.5, 0), "M2":(4.7, 0),
    "S1":(0.5, h), "S2":(1.5, h), "S3":(2.5, h), "S4":(3.5, h), "S5":(4.5, h), "S6":(5.5, h), "S7":(6.5, h),
    "Z1":(1, 2*h), "Z2":(2, 2*h), "Z3":(3.5, 2*h), "Z4":(5, 2*h), "Z5":(6, 2*h),
}

# Add colors for nodes
colors = ["#66b3e7", "#fde397", "#d9958f"]
for node, layer in G.nodes(data='layer'):
    if layer is not None:
        G.nodes[node]['color'] = colors[layer]

nx.draw_networkx_edges(G, pos, edge_color="#568ec2", width=2)
nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[node]['color'] for node in G.nodes], node_size=1500)
nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_color='black', 
                        font_size=20, font_weight='bold')

'''
for node in G.nodes:
    plt.gca().add_patch(patches.Rectangle((pos[node][0]-0.2, pos[node][1]-0.1), 0.4, 0.2, 
                                          facecolor=G.nodes[node]['color']))
    plt.gca().text(pos[node][0], pos[node][1], s=node, color='black', 
                   verticalalignment='center', horizontalalignment='center')
'''

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              label='minimum' if i == 2 else f'index-{2-i}', 
                              markerfacecolor=color, markersize=10) 
                   for i, color in enumerate(colors)]
plt.legend(handles=legend_elements, frameon=False, fontsize=12)

plt.xlim(min(pos.values(), key=lambda x: x[0])[0]-0.5, max(pos.values(), key=lambda x: x[0])[0]+0.5)
plt.ylim(min(pos.values(), key=lambda x: x[1])[1]-0.5, max(pos.values(), key=lambda x: x[1])[1]+0.5)

plt.gca().invert_yaxis()
plt.axis('off')
# plt.savefig("Tree2.png", dpi=300)
plt.show()