import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import logging
from sklearn.metrics import silhouette_score
from networkx.algorithms.community.quality import modularity

# 设置日志记录配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='/home/599/xs5813/4880/pattern_detection/l_result/community_detection_Louvain.log')

# 加载图
logging.info("Loading the graph from '/scratch/iu60/xs5813/graph_aus_mask_net.graphml'.")
G = nx.read_graphml("/scratch/iu60/xs5813/graph_Aus_mask_3year_thresh2_same_t_and_r.graphml")

# 使用Louvain算法进行社区检测
logging.info("Applying Louvain algorithm for community detection.")
partition = community_louvain.best_partition(G)

# 获取社区数量
n_clusters = len(set(partition.values()))
# 初始化社区的经纬度数组
communities_positions = [[] for _ in range(n_clusters)]
communities_nodes = [[] for _ in range(n_clusters)]  # 用于存储每个社区的节点
# 获取每个社区的节点位置信息并记录到日志
for node, label in partition.items():
    pos_str = G.nodes[node]['pos']  # 获取位置字符串
    lat, lon = map(float, pos_str.split(","))  # 假设经纬度以"lat,lon"的形式存储，并转换为浮点数
    communities_positions[label].append((lat, lon))  # 将经纬度元组添加到相应的社区列表
    communities_nodes[label].append(node)  # 将节点添加到相应的社区列表

# 将每个社区的位置信息保存到文件
for i, positions in enumerate(communities_positions):
    filename = f"/home/599/xs5813/4880/pattern_detection/l_result/community_{i+1}_positions.npy"
    np.save(filename, positions)  # 保存为.npy文件
    logging.info(f"Community {i+1} positions saved to {filename}")

# 打印确认消息
print("All community positions have been saved to the current directory.")

# 模块度（Modularity）
mod_value = modularity(G, communities_nodes)
logging.info(f"Modularity: {mod_value}")

# 轮廓系数（Silhouette Coefficient）
# 创建节点标签数组
node_labels = np.array([partition[node] for node in G.nodes()])
# 创建节点特征数组（这里使用节点的度作为特征）
node_features = np.array([[G.degree(node)] for node in G.nodes()])
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
# 打印评估结果确认消息
print("Community detection evaluation metrics have been logged.")
