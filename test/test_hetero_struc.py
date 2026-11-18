import torch
# Load heterogeneous graph dataset

data = torch.load("data/3.function/function-data_CompSimilarities0.85_PPI_Score460.pt")

# Print graph structure information
print("Graph structure:")
print(data)

# Print node information
print("\nNode information:")
for node_type in data.node_types:
    print(f"{node_type}: {data[node_type].num_nodes} nodes")

# Print edge information
print("\nEdge information:")
for edge_type in data.edge_types:
    print(f"{edge_type}: {data[edge_type].edge_index.size(1)} edges")

# Print other metadata
print("\nOther metadata:")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
print(f"Has self loops: {data.has_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")
