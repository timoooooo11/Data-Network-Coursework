"""import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from libpysal.weights import Queen
from esda.moran import Moran
from scipy.spatial import distance_matrix


#Download the road network for a 1 km² area around Leeds city center (driveable roads only)
leeds_centre = (53.7996, -1.5491)
G = ox.graph_from_point(leeds_centre, dist=1000, network_type='drive')

#Compute network characteristics
stats = ox.basic_stats(G)
largest_cc = max(nx.strongly_connected_components(G), key=len)
subgraph = G.subgraph(largest_cc)
diameter = nx.diameter(nx.Graph(subgraph))
circuitry = stats["edge_length_total"] / stats["street_length_avg"] - 1
num_nodes = stats["n"]
num_intersections = stats["intersection_count"]
area_km2 = (2 * 1000) ** 2 / 1e6
node_density = num_nodes / area_km2
intersection_density = num_intersections / area_km2

network_results = {
    "Spatial Diameter": diameter,
    "Average Street Length": stats["street_length_avg"],
    "Node Density": node_density,  
    "Intersection Density": intersection_density, 
    "Edge Density": stats["street_segment_count"] / area_km2,  
    "Circuitry": circuitry
}
print("Network Analysis Results:", network_results)

#Check if the road network is planar
planarity = nx.check_planarity(nx.Graph(G))
print("Is the network planar?", planarity[0])

#Load accident data
file_path = "/Users/wangyining/Desktop/kcl/data_network/coursework/mydata/Traffic_accidents_2019_Leeds.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip() #Process column names to remove possible Spaces
data["geometry"] = data.apply(lambda row: Point(row["Grid Ref: Easting"], row["Grid Ref: Northing"]), axis=1) #Convert the incident data to GeoDataFrame
accidents_gdf = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:27700") #UK National grid coordinates

#Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ox.plot_graph(G, ax=ax, node_size=10, node_color="red", edge_color="gray", show=False)
accidents_gdf.plot(ax=ax, color="blue", markersize=20, label="Accident")
plt.legend()
plt.title("Accident Locations on Leeds Road Network")
plt.show()

#Compute Moran’s I for spatial correlation
if "Number of Vehicles" in accidents_gdf.columns:
    w = Queen.from_dataframe(accidents_gdf)
    moran = Moran(accidents_gdf["Number of Vehicles"], w)
    print("Moran's I:", moran.I, "p-value:", moran.p_norm)
else:
    print("Error: 'Number of Vehicles' column not found in the data.")

# Compute k-function (spatial clustering analysis)
def k_function(points, max_dist):
    distances = distance_matrix(points, points)
    return np.sum(distances < max_dist) / (len(points) * (len(points) - 1))
accident_points = np.array(list(zip(data["Grid Ref: Easting"], data["Grid Ref: Northing"])))
k_value = k_function(accident_points, 500) #Compute accident clustering within 500 meters
print("K-function result:", k_value)

#Compute distance from accidents to the nearest intersection
nodes, edges = ox.graph_to_gdfs(G)
nodes = nodes.to_crs(epsg=27700)  #Ensure coordinate system matches
def nearest_node(point):
    distances = nodes.geometry.distance(point)
    nearest_geom = distances.idxmin()
    return nodes.loc[nearest_geom].geometry
data["Nearest Intersection"] = data["geometry"].apply(nearest_node)
data["Distance to Intersection"] = data.apply(lambda row: row["geometry"].distance(row["Nearest Intersection"]), axis=1)

#Visualize distribution of accident distances to intersections
plt.hist(data["Distance to Intersection"], bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.xlabel("Distance to Nearest Intersection (meters)")
plt.ylabel("Number of Accidents")
plt.title("Distribution of Accident Distances to Intersections")
plt.show()
"""
import os
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from libpysal.weights import Queen
from esda.moran import Moran
from scipy.spatial import distance_matrix

# 创建图像输出文件夹
os.makedirs("figures", exist_ok=True)

# Leeds 中心 1km² 区域道路网（只取 drive 路网）
leeds_centre = (53.7996, -1.5491)
G = ox.graph_from_point(leeds_centre, dist=1000, network_type='drive')

# 只保留最大无向连通子图用于 diameter 计算
G_undirected = G.to_undirected()
largest_cc = max(nx.connected_components(G_undirected), key=len)
G_largest = G_undirected.subgraph(largest_cc).copy()

# 计算网络指标
stats = ox.basic_stats(G)
diameter = nx.diameter(G_largest)
circuitry = stats["edge_length_total"] / stats["street_length_avg"] - 1
area_km2 = (2 * 1000) ** 2 / 1e6
network_results = {
    "Spatial Diameter": diameter,
    "Average Street Length": stats["street_length_avg"],
    "Node Density": stats["n"] / area_km2,
    "Intersection Density": stats["intersection_count"] / area_km2,
    "Edge Density": stats["street_segment_count"] / area_km2,
    "Circuitry": circuitry
}
print("Network Analysis Results:", network_results)

# 判断是否为平面图
is_planar = nx.check_planarity(G_undirected)[0]
print("Is the network planar?", is_planar)

# 导入事故数据
file_path = "/Users/wangyining/Desktop/kcl/data_network/coursework/mydata/Traffic_accidents_2019_Leeds.csv" 
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()
data["geometry"] = data.apply(lambda row: Point(row["Grid Ref: Easting"], row["Grid Ref: Northing"]), axis=1)
accidents_gdf = gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:27700")

# 事故点叠加路网图
fig, ax = plt.subplots(figsize=(10, 10))
ox.plot_graph(G, ax=ax, node_size=10, node_color="red", edge_color="gray", show=False)
accidents_gdf.plot(ax=ax, color="blue", markersize=20, label="Accident")
plt.legend()
plt.title("Accident Locations on Leeds Road Network")
plt.savefig("figures/accident_map.png")
plt.close()

# Moran’s I 空间相关性分析
if "Number of Vehicles" in accidents_gdf.columns:
    w = Queen.from_dataframe(accidents_gdf)
    moran = Moran(accidents_gdf["Number of Vehicles"], w)
    print("Moran's I:", moran.I, "p-value:", moran.p_norm)
else:
    print("Column 'Number of Vehicles' not found.")

# k-function 曲线（不同距离下的空间聚集性）
def k_function(points, max_dist):
    distances = distance_matrix(points, points)
    return np.sum(distances < max_dist) / (len(points) * (len(points) - 1))

accident_points = np.array(list(zip(data["Grid Ref: Easting"], data["Grid Ref: Northing"])))
r_values = np.arange(100, 2000, 100)
k_results = [k_function(accident_points, r) for r in r_values]

plt.plot(r_values, k_results, marker='o')
plt.xlabel("Radius (m)")
plt.ylabel("K-function value")
plt.title("K-function Curve for Accident Clustering")
plt.grid(True)
plt.savefig("figures/k_function_curve.png")
plt.close()

# 计算每个事故到最近交叉点的距离
nodes, _ = ox.graph_to_gdfs(G)
nodes = nodes.to_crs(epsg=27700)

def nearest_node(point):
    distances = nodes.geometry.distance(point)
    return nodes.loc[distances.idxmin()].geometry

data["Nearest Intersection"] = data["geometry"].apply(nearest_node)
data["Distance to Intersection"] = data.apply(lambda row: row["geometry"].distance(row["Nearest Intersection"]), axis=1)

# 输出平均距离和中位数
mean_dist = data["Distance to Intersection"].mean()
median_dist = data["Distance to Intersection"].median()
print("Mean distance to intersection:", mean_dist)
print("Median distance to intersection:", median_dist)

# 可视化：事故距离分布直方图
plt.figure(figsize=(10, 6))
plt.hist(data["Distance to Intersection"], bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.axvline(mean_dist, color='red', linestyle='--', label=f"Mean = {mean_dist:.2f} m")
plt.axvline(median_dist, color='green', linestyle='--', label=f"Median = {median_dist:.2f} m")
plt.xlabel("Distance to Nearest Intersection (meters)")
plt.ylabel("Number of Accidents")
plt.title("Distribution of Accident Distances to Intersections")
plt.legend()
plt.tight_layout()
plt.savefig("figures/distance_to_intersection_hist.png")
plt.close()





