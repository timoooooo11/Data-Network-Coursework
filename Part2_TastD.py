import os
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from pykeen.pipeline import pipeline
from pykeen.datasets import CoDExMedium
import matplotlib.pyplot as plt

# ============ Part D.1: PageRank ============
print("=== PageRank Analysis ===")
center = (53.7996, -1.5491)
G = ox.graph_from_point(center, dist=6000, network_type='walk')
directed_G = nx.DiGraph(G)
pagerank_scores = nx.pagerank(directed_G, alpha=0.85)
pagerank_df = pd.DataFrame(pagerank_scores.items(), columns=["Node", "PageRank"]).sort_values("PageRank", ascending=False)
print(pagerank_df.head(10))


# ============ Part D.2: Knowledge Graph Embeddings ============
print("\n=== TransE and RotatE Embeddings on CoDEx-Medium ===")
transE_result = pipeline(
    model="TransE",
    dataset=CoDExMedium(),
    training_kwargs=dict(num_epochs=100, batch_size=256),
)
print("✅ TransE Mean Rank:", transE_result.get_metric("mean_rank"))

rotate_result = pipeline(
    model="RotatE",
    dataset=CoDExMedium(),
    training_kwargs=dict(num_epochs=100, batch_size=256),
)
print("✅ RotatE Mean Rank:", rotate_result.get_metric("mean_rank"))


# ============ Part D.3: GCN on Road Graph ============
print("\n=== GCN on Road Graph ===")
G = nx.convert_node_labels_to_integers(G, label_attribute="old_id")
node_mapping = {old: new for new, old in enumerate(G.nodes())}
edge_list = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
num_nodes = len(G.nodes())
x = torch.rand((num_nodes, 16))  # random node features

# Check edge validity
assert edge_index.max().item() < num_nodes

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Training GCN...")
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, torch.rand((num_nodes, 2)))
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("✅ GCN Training Completed")