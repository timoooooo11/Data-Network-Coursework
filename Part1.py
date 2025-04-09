import pandas as pd
import networkx as nx
import numpy as np
import EoN
import matplotlib.pyplot as plt
from collections import Counter
import os

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 创建保存图像文件夹
os.makedirs("figures", exist_ok=True)

# 数据读取路径
admin_df = pd.read_csv('/Users/wangyining/Desktop/kcl/data_network/coursework/datasets/ADMINISTRATORS.csv', delimiter=',')
prop_df = pd.read_csv('/Users/wangyining/Desktop/kcl/data_network/coursework/datasets/PROPERTY_PROPOSAL.csv', delimiter=',')
users_df = pd.read_csv('/Users/wangyining/Desktop/kcl/data_network/coursework/datasets/USERS.csv', delimiter=',')


# 统一列名
columns = ['thread_subject', 'username', 'page_name']
admin_df.columns = columns
prop_df.columns = columns
users_df.columns = columns

# 网络构建函数
def build_network(df, name):
    G = nx.Graph()
    for (page, thread), group in df.groupby(['page_name', 'thread_subject']):
        users = group['username'].unique()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                G.add_edge(users[i], users[j])
    print(f"Network {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# 网络构建
G_admin = build_network(admin_df, 'ADMINISTRATORS')
G_prop = build_network(prop_df, 'PROPERTY_PROPOSAL')
G_users = build_network(users_df, 'USERS')

# 网络分析函数
def analyze_network(G, name):
    print(f"\nAnalyzing {name} network")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")

    # 保留最大连通分量用于后续分析
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    degrees = [deg for node, deg in G.degree()]
    avg_degree = np.mean(degrees)
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Diameter: {nx.diameter(G)}")
    print(f"Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
    print(f"Clustering coefficient: {nx.average_clustering(G):.2f}")

    # 度分布可视化（log-log 图）
    degree_counts = Counter(degrees)
    degrees_sorted, counts_sorted = zip(*sorted(degree_counts.items()))
    plt.figure(figsize=(8, 6))
    plt.loglog(degrees_sorted, counts_sorted, 'bo')
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Degree Distribution of {name} Network', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"figures/degree_distribution_{name.lower()}.png")
    plt.close()

# 分析三张图
analyze_network(G_admin, 'ADMINISTRATORS')
analyze_network(G_prop, 'PROPERTY_PROPOSAL')
analyze_network(G_users, 'USERS')

# SIR 模拟函数
def simulate_sir(G, name):
    print(f"\nSimulating SIR model on {name} network...")

    # 保留最大连通分量
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    beta = 0.1
    gamma = 0.02
    initial_infected = set(np.random.choice(list(G.nodes), size=5, replace=False))
    SIR = EoN.fast_SIR(G, beta, gamma, initial_infecteds=initial_infected)

    # 绘图
    t, S, I, R = SIR
    plt.figure(figsize=(8, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.title(f'SIR Model Simulation on {name} Network', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/sir_simulation_{name.lower()}.png")
    plt.close()

# SIR 模拟
simulate_sir(G_admin, 'ADMINISTRATORS')
simulate_sir(G_prop, 'PROPERTY_PROPOSAL')
simulate_sir(G_users, 'USERS')




