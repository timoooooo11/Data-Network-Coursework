import os
import random
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import voronoi_diagram
import pykeen
from pykeen.pipeline import pipeline
from pykeen.datasets import CoDExMedium


# 输出目录
os.makedirs("figures", exist_ok=True)

# 1. 下载扩大范围的 Leeds 路网
leeds_center = (53.7996, -1.5491)
G = ox.graph_from_point(leeds_center, dist=6000, network_type='drive')
nodes, edges = ox.graph_to_gdfs(G)
nodes = nodes.to_crs(epsg=27700)

# 2. 随机选择 2 个起点
np.random.seed(42)
sampled_nodes = nodes.sample(4)
seed_points = gpd.GeoSeries(sampled_nodes.geometry.values, crs="EPSG:27700")

# 3. 构建 Voronoi 区域（shapely + convex hull 裁剪）
leeds_union = nodes.geometry.unary_union
leeds_boundary = leeds_union.convex_hull
vd = voronoi_diagram(seed_points.unary_union, envelope=leeds_boundary, tolerance=0.001)
voronoi_polygons = [poly.intersection(leeds_boundary) for poly in vd.geoms if poly.is_valid and not poly.is_empty]
voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_polygons, crs="EPSG:27700")
print(f"Number of valid Voronoi cells: {len(voronoi_gdf)}")

# 4. 搜索每个 Voronoi cell 中约 42km 路线
marathon_routes = []
for polygon in voronoi_gdf.geometry:
    nodes_in_cell = nodes[nodes.geometry.within(polygon)]
    if len(nodes_in_cell) < 10:
        continue

    subgraph = G.subgraph(nodes_in_cell.index).to_undirected()
    node_list = list(subgraph.nodes)
    found = False

    for i in range(min(20, len(node_list))):
        for j in range(i + 1, min(40, len(node_list))):
            src = node_list[i]
            dst = node_list[j]

            try:
                d = nodes.loc[src].geometry.distance(nodes.loc[dst].geometry)
                if d < 500:
                    continue
            except:
                continue

            try:
                path = nx.shortest_path(subgraph, src, dst, weight='length')
                reversed_path = path[::-1][1:]
                path_cycle = path + reversed_path
                edge_lengths = ox.utils_graph.get_route_edge_attributes(subgraph, path_cycle, 'length')
                total_length = sum(edge_lengths)
                if 35000 <= total_length <= 47000:
                    marathon_routes.append((path_cycle, total_length))
                    found = True
                    break
            except:
                continue
        if found:
            break

print(f"Found {len(marathon_routes)} valid marathon routes.")

# 5. 可视化绘图
edges = edges.to_crs(epsg=27700)
sampled_nodes = sampled_nodes.to_crs(epsg=27700)
voronoi_gdf = voronoi_gdf.to_crs(epsg=27700)

fig, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, linewidth=0.5, color="lightgray")
if not voronoi_gdf.empty:
    voronoi_gdf.boundary.plot(ax=ax, linewidth=1.2, color="black", label="Voronoi Boundaries")
sampled_nodes.plot(ax=ax, color="red", markersize=100, label="Start Points")

for (cycle, length) in marathon_routes:
    path_coords = [nodes.loc[n].geometry for n in cycle if n in nodes.index]
    if len(path_coords) > 1:
        gpd.GeoSeries([LineString(path_coords)], crs=nodes.crs).plot(ax=ax, color='blue', linewidth=2, label="42km Route")

# 设置缩放边界
buffer = 300
xmin, ymin, xmax, ymax = nodes.total_bounds
ax.set_xlim(xmin - buffer, xmax + buffer)
ax.set_ylim(ymin - buffer, ymax + buffer)

# 去重图例
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='lower left')

plt.title("Voronoi Cells and Marathon Routes (Reliable Version)")
plt.tight_layout()
plt.savefig("figures/voronoi_marathon_routes_final.png")
plt.close()
