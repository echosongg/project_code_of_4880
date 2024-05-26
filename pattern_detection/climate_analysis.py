import numpy as np
import logging
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
def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 设置日志配置
logging.basicConfig(level=logging.INFO, filename='community_averages.log', filemode='w', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

loaded_train_data = load_data("/scratch/iu60/xs5813/train_data_small_Aus.pkl")

# 检查 loaded_train_data 的类型和内容
for attr in ['precip', 'tmax', 'tmin']:
    if hasattr(loaded_train_data, attr):
        data = getattr(loaded_train_data, attr)
        logging.info(f"属性名: {attr}, 形状: {np.shape(data)}")

# 提取对应的数据
precip_data = np.array(getattr(loaded_train_data, 'precip', None))
tmax_data = np.array(getattr(loaded_train_data, 'tmax', None))
tmin_data = np.array(getattr(loaded_train_data, 'tmin', None))
def lat_lon_to_index(lat, lon, lat_range=(-44.525, -9.975), lon_range=(111.975, 156.275), lat_size=None, lon_size=None):
    """
    将给定的纬度和经度转换为数据数组的索引值。

    参数：
        lat (float or array): 纬度值。
        lon (float or array): 经度值。
        lat_range (tuple): 纬度范围，默认为 (-44.525, -9.975)。
        lon_range (tuple): 经度范围，默认为 (111.975, 156.275)。
        lat_size (int): 纬度方向的数据点数量。如果为 None，则从 `precip_data` 的形状中推断。
        lon_size (int): 经度方向的数据点数量。如果为 None，则从 `precip_data` 的形状中推断。

    返回：
        tuple: 包含纬度索引和经度索引的元组。
    """
    if lat_size is None or lon_size is None:
        raise ValueError("lat_size 和 lon_size 不能为空")

    # 计算纬度和经度步长
    lat_step = (lat_range[1] - lat_range[0]) / (lat_size)
    lon_step = (lon_range[1] - lon_range[0]) / (lon_size)

    # 计算索引值
    lat_index = np.round((lat - lat_range[0]) / lat_step).astype(int)
    lon_index = np.round((lon - lon_range[0]) / lon_step).astype(int)
    logging.info(f"纬度: {lat}, 经度: {lon}, 纬度索引: {lat_index}, 经度索引: {lon_index}")

    return lat_index, lon_index

community_positions = []
for i in range(1, 6):
    path = f'/home/599/xs5813/4880/pattern_detection/l_result/community_{i}_positions.npy'
    community_positions.append(np.load(path))
lat_size, lon_size = precip_data.shape[1], precip_data.shape[2]
# 计算并打印每个社区的平均降雨量、最高气温和最低气温
for index, positions in enumerate(community_positions, 1):
    if positions.size > 0:  # 检查是否有位置数据

        # 获取索引值
        lat, lon = positions[:, 0], positions[:, 1]
        lat, lon = lat_lon_to_index(lat, lon, lat_size=lat_size, lon_size=lon_size)
        
        # 根据索引提取相应位置的数据
        community_precip = precip_data[lat, lon]
        community_tmax = tmax_data[lat, lon]
        community_tmin = tmin_data[lat, lon]
        
        # 计算平均值
        avg_precip = np.mean(community_precip)
        avg_tmax = np.mean(community_tmax)
        avg_tmin = np.mean(community_tmin)
        
        # 打印结果
        logging.info(f"Community {index}: Avg Precip = {avg_precip:.2f} mm, Avg Tmax = {avg_tmax:.2f} °C, Avg Tmin = {avg_tmin:.2f} °C")