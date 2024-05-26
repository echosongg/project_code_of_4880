import numpy as np
import networkx as nx
import logging
from sklearn.metrics import silhouette_score
from networkx.algorithms.community.quality import modularity

# 设置日志记录配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='/home/599/xs5813/4880/pattern_detection/p_result/community_detection_propagation.log')

# 加载图
logging.info("Loading the graph from 'graph_Aus_mask_3year_thresh2_same_t_and_r.graphml'.")
G = nx.read_graphml("/scratch/iu60/xs5813/graph_Aus_mask_3year_thresh2_same_t_and_r.graphml")

# 使用标签传播算法进行社区检测
logging.info("Applying Label Propagation algorithm for community detection.")
community_generator = nx.algorithms.community.label_propagation_communities(G)
labels = list(community_generator)  # 将生成器转换为列表

# 获取社区数量
n_clusters = len(labels)

# 初始化社区的经纬度数组和节点数组
communities_positions = [[] for _ in range(n_clusters)]
communities_nodes = [[] for _ in range(n_clusters)]  # 用于存储每个社区的节点

# 将标签传播算法返回的结果转换为节点和标签的映射
label_mapping = {}
for idx, community in enumerate(labels):
    for node in community:
        label_mapping[node] = idx
        communities_nodes[idx].append(node)  # 将节点添加到相应的社区列表

# 获取每个社区的节点位置信息并记录到日志
for node, label in label_mapping.items():
    pos_str = G.nodes[node]['pos']  # 获取位置字符串
    lat, lon = map(float, pos_str.split(","))  # 假设经纬度以"lat,lon"的形式存储，并转换为浮点数
    communities_positions[label].append((lat, lon))  # 将经纬度元组添加到相应的社区列表

# 将每个社区的位置信息保存到文件
for i, positions in enumerate(communities_positions):
    filename = f"/home/599/xs5813/4880/pattern_detection/p_result/community_{i+1}_positions.npy"
    np.save(filename, positions)  # 保存为.npy文件
    logging.info(f"Community {i+1} positions saved to {filename}")

# 打印确认消息
print("All community positions have been saved to the current directory.")

# 评估社区检测的质量

# 模块度（Modularity）
mod_value = modularity(G, communities_nodes)
logging.info(f"Modularity: {mod_value}")

# 轮廓系数（Silhouette Coefficient）
# 创建节点标签数组
node_labels = np.array([label_mapping[node] for node in G.nodes()])
# 创建节点特征数组（这里使用节点的度作为特征）
node_features = np.array([[G.degree(node)] for node in G.nodes()])
#silhouette_avg = silhouette_score(node_features, node_labels)
#logging.info(f"Silhouette Coefficient: {silhouette_avg}")

# 平均内部度（Average Internal Degree）
def average_internal_degree(G, communities_nodes):
    avg_internal_degrees = []
    for nodes in communities_nodes:
        subgraph = G.subgraph(nodes)
        internal_edges = subgraph.number_of_edges()
        num_nodes = subgraph.number_of_nodes()
        if num_nodes > 0:
            avg_internal_degree = (2 * internal_edges) / num_nodes
            avg_internal_degrees.append(avg_internal_degree)
        else:
            avg_internal_degrees.append(0)
    return avg_internal_degrees

avg_internal_degrees = average_internal_degree(G, communities_nodes)
for i, avg_internal_degree in enumerate(avg_internal_degrees):
    logging.info(f"Community {i+1} Average Internal Degree: {avg_internal_degree}")

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


