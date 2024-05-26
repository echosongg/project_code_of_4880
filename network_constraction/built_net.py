import os
import pickle
import numpy as np
import networkx as nx
import itertools
import torch
import logging
import matplotlib.pyplot as plt
from net_cons import AWAP
from sklearn.preprocessing import StandardScaler

# 设置日志记录
logging.basicConfig(filename='graph_mask_built.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 定义标准化函数
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)

loaded_train_data = load_data("/scratch/iu60/xs5813/Australian_inland.pkl")

# 检查 loaded_train_data 的类型和内容
for attr in ['precip', 'tmax', 'tmin']:
    if hasattr(loaded_train_data, attr):
        data = getattr(loaded_train_data, attr)
        logging.info(f"属性名: {attr}, 形状: {np.shape(data)}")

# 提取对应的数据
precip_data = np.array(getattr(loaded_train_data, 'precip', None))
tmax_data = np.array(getattr(loaded_train_data, 'tmax', None))
tmin_data = np.array(getattr(loaded_train_data, 'tmin', None))

# 对数据进行标准化
precip_data_std = standardize_data(precip_data)
tmax_data_std = standardize_data(tmax_data)
tmin_data_std = standardize_data(tmin_data)

# 将数据传输到 GPU 上
precip_data = torch.tensor(precip_data_std, device=device)
tmax_data = torch.tensor(tmax_data_std, device=device)
tmin_data = torch.tensor(tmin_data_std, device=device)


# 定义气象相似度计算函数
def calculate_similarity(precip1, precip2, tmax1, tmax2, tmin1, tmin2):
    return torch.sqrt(torch.mean((precip1 - precip2) ** 2) + 
                     (torch.mean((tmax1 - tmax2) ** 2) + 
                      torch.mean((tmin1 - tmin2) ** 2)))

# 定义综合权重计算函数
def calculate_weight(similarity, alpha=1, beta=1, epsilon=1e-5):
    return 1 / (similarity + epsilon)

# 获取经纬度信息
lat_size, lon_size = precip_data.shape[1], precip_data.shape[2]
lats = np.linspace(-44.525, -9.975, lat_size)
lons = np.linspace(111.975, 156.275, lon_size)
lat_lon_pairs = list(itertools.product(lats, lons))

# 初始化图
G = nx.Graph()

# 添加节点，并将元组 (lat, lon) 转换为字符串
for i, (lat, lon) in enumerate(lat_lon_pairs):
    G.add_node(i, pos=f"{lat},{lon}")

pair_count = 0  # 初始化计数器
threshold = 2
for (i, (lat1, lon1)), (j, (lat2, lon2)) in itertools.combinations(enumerate(lat_lon_pairs), 2):
    # 获取当前节点的数据
    if torch.isnan(precip_data[:, i // lon_size, i % lon_size]).any() or torch.isnan(precip_data[:, j // lon_size, j % lon_size]).any():
        continue  # Skip this iteration if any data point is NaN, which mean it is ocean.
    precip1 = precip_data[:, i // lon_size, i % lon_size]
    tmax1 = tmax_data[:, i // lon_size, i % lon_size]
    tmin1 = tmin_data[:, i // lon_size, i % lon_size]

    precip2 = precip_data[:, j // lon_size, j % lon_size]
    tmax2 = tmax_data[:, j // lon_size, j % lon_size]
    tmin2 = tmin_data[:, j // lon_size, j % lon_size]

    # 计算标准化后的距离和气象相似度
    #distance = calculate_distance(lat1, lon1, lat2, lon2)
    similarity = calculate_similarity(precip1, precip2, tmax1, tmax2, tmin1, tmin2)

    # 计算综合权重
    weight = calculate_weight(similarity).item()  # 使用 .item() 将 torch.Tensor 转换为 float

    # 只添加权重低于阈值的边
    if similarity < 2:
        G.add_edge(i, j, weight=weight)
    
    # 每处理5000对节点记录一次日志
    pair_count += 1
    if pair_count % 5000 == 0:
            # 记录 similarity 和 weight 的值
        logging.info(f"已处理 {pair_count} 对节点, Similarity between ({i}, {j}): {similarity} weight {weight}")

node_count = G.number_of_nodes()
edge_count = G.number_of_edges()

# 计算平均度数
average_degree = sum(dict(G.degree()).values()) / node_count

# 将这些信息记录到日志中
logging.info(f"Total number of nodes: {node_count}")
logging.info(f"Total number of edges: {edge_count}")
logging.info(f"Average degree of nodes: {average_degree:.2f}")

# 保存图为 GraphML 文件
nx.write_graphml(G, "/scratch/iu60/xs5813/graph_Aus_mask_3year_thresh2_same_t_and_r.graphml.graphml")
logging.info("Graph saved as 'graph_Aus'")
