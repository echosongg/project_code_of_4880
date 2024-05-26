# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from sklearn.cluster import SpectralClustering
# import logging

# # 设置日志记录配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='community_detection.log')

# # 加载图
# logging.info("Loading the graph from 'graph_creation_Aus_no_dis.graphml'.")
# G = nx.read_graphml("../network_constraction/graph_Aus_no_dis.graphml")

# # 将图转换为邻接矩阵
# logging.info("Converting the graph to an adjacency matrix.")
# adj_matrix = nx.to_numpy_array(G)

# # 定义谱聚类模型
# n_clusters = 6  # 根据需要调整社区数量
# logging.info(f"Defining Spectral Clustering model with {n_clusters} clusters.")
# spectral_model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')

# # 应用谱聚类
# logging.info("Applying spectral clustering.")
# labels = spectral_model.fit_predict(adj_matrix)

# # 初始化四个社区的经纬度数组
# communities_positions = [[] for _ in range(n_clusters)]

# # 获取每个社区的节点位置信息并记录到日志
# for node, label in zip(G.nodes(), labels):
#     pos_str = G.nodes[node]['pos']  # 获取位置字符串
#     lat, lon = map(float, pos_str.split(","))  # 假设经纬度以"lat,lon"的形式存储，并转换为浮点数
#     communities_positions[label].append((lat, lon))  # 将经纬度元组添加到相应的社区列表

# # 将每个社区的位置信息保存到文件
# for i, positions in enumerate(communities_positions):
#     filename = f"community_{i+1}_positions.npy"
#     np.save(filename, positions)  # 保存为.npy文件
#     logging.info(f"Community {i+1} positions saved to {filename}")

# # 打印确认消息
# print("All community positions have been saved to the current directory.")

import numpy as np
import networkx as nx
import logging
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from networkx.algorithms.community.quality import modularity

# 设置日志记录配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='/home/599/xs5813/4880/pattern_detection/s_result/community_detection_spectral.log')

# 加载图
logging.info("Loading the graph from '/scratch/iu60/xs5813/graph_Aus_mask_3year_thresh2_same_t_and_r.graphml'.")
G = nx.read_graphml("/scratch/iu60/xs5813/graph_Aus_mask_3year_thresh2_same_t_and_r.graphml")

# 将图转换为邻接矩阵
logging.info("Converting the graph to an adjacency matrix.")
adj_matrix = nx.to_numpy_array(G)

# 寻找最佳社区数量
logging.info("Searching for the optimal number of clusters.")
modularity_scores = []
range_n_clusters = range(2, 10)  # 可以调整这个范围来找到最佳的簇数
best_labels = None
for n_clusters in range_n_clusters:
    spectral_model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
    labels = spectral_model.fit_predict(adj_matrix)
    community_list = [[] for _ in range(n_clusters)]
    for node, label in zip(G.nodes(), labels):
        community_list[label].append(node)
    mod_value = modularity(G, community_list)
    modularity_scores.append(mod_value)
    logging.info(f"Number of clusters: {n_clusters}, Modularity: {mod_value}")
    if mod_value == max(modularity_scores):
        best_labels = labels

# 选择模块度最高的簇数
optimal_n_clusters = range_n_clusters[np.argmax(modularity_scores)]
logging.info(f"Optimal number of clusters: {optimal_n_clusters}")

# 使用最佳簇数重新运行谱聚类
spectral_model = SpectralClustering(n_clusters=optimal_n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
labels = spectral_model.fit_predict(adj_matrix)

# 初始化社区的经纬度数组
communities_positions = [[] for _ in range(optimal_n_clusters)]
communities_nodes = [[] for _ in range(optimal_n_clusters)]  # 新增这行

# 获取每个社区的节点位置信息并记录到日志
for node, label in zip(G.nodes(), labels):
    pos_str = G.nodes[node]['pos']
    lat, lon = map(float, pos_str.split(","))
    communities_positions[label].append((lat, lon))

# 将每个社区的位置信息保存到文件
for i, positions in enumerate(communities_positions):
    filename = f"/home/599/xs5813/4880/pattern_detection/s_result/community_{i+1}_positions.npy"
    np.save(filename, positions)
    logging.info(f"Community {i+1} positions saved to {filename}")

# 打印确认消息
print("All community positions have been saved to the current directory.")

# 评估社区检测的质量

# 模块度（Modularity）
community_list = [[] for _ in range(optimal_n_clusters)]
for node, label in zip(G.nodes(), labels):
    community_list[label].append(node)
mod_value = modularity(G, community_list)
logging.info(f"Modularity: {mod_value}")

# 轮廓系数（Silhouette Coefficient）
node_labels = np.array([labels[node] for node in range(len(G.nodes()))])
node_features = np.array([[adj_matrix[i][j] for j in range(len(adj_matrix[i]))] for i in range(len(adj_matrix))])
silhouette_avg = silhouette_score(node_features, node_labels)
logging.info(f"Silhouette Coefficient: {silhouette_avg}")

def calculate_global_efficiency(G):
    n = len(G)
    dist_sum = 0
    for node in G:
        path_length = nx.single_source_shortest_path_length(G, node)
        dist_sum += sum([1/p if p > 0 else 0 for p in path_length.values()])
    return dist_sum / (n * (n - 1))

# 计算全局效率
global_eff = calculate_global_efficiency(G)
logging.info(f"Global Efficiency: {global_eff}")

# 计算Conductance
def calculate_conductance(G, communities_nodes):
    conductance_list = []
    for nodes in communities_nodes:
        cut_size = nx.cut_size(G, nodes)
        volume = sum([G.degree(n) for n in nodes])
        if volume == 0 or 2 * G.size() - volume == 0:  # Checking if the denominator could be zero
            conductance = 0
        else:
            conductance = cut_size / min(volume, 2 * G.size() - volume)
        conductance_list.append(conductance)
    return sum(conductance_list) / len(conductance_list) if conductance_list else 0

conductance_avg = calculate_conductance(G, communities_nodes)
logging.info(f"Conductance: {conductance_avg}")


