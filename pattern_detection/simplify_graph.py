import networkx as nx
import numpy as np
import logging
from mpl_toolkits.basemap import Basemap

# 设置日志
logging.basicConfig(filename='simplify_graph.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 加载图
G = nx.read_graphml("../network_constraction/graph_Aus_no_dis.graphml")
logging.info("Graph loaded from 'graph_Aus_no_dis.graphml'")
lon_min, lon_max = 111.975, 156.275
lat_min, lat_max = -44.525, -9.975
# 创建 Basemap 实例
# 创建 Basemap 实例，精确匹配您的数据范围
m = Basemap(projection='merc', llcrnrlat=-44.525, urcrnrlat=-9.975, llcrnrlon=111.975, urcrnrlon=156.275, resolution='i')

# 提取每个节点的经纬度，并检查是否在海洋中
nodes_to_remove = []
for node, data in G.nodes(data=True):
    lon, lat = map(float, data['pos'].split(','))
    if m.is_land(lon, lat):
        logging.info(f"Node {node} is on land at position {lon}, {lat}.")
    else:
        logging.info(f"Node {node} is in the ocean at position {lon}, {lat}.")
        nodes_to_remove.append(node)

# 删除位于海洋的节点
for node in nodes_to_remove:
    G.remove_node(node)
    logging.info(f"Removed node {node} because it is located in the ocean.")

# 保存简化后的图
nx.write_graphml(G, "../network_constraction/simplified_graph_Aus_no_dis.graphml")
logging.info("Simplified graph saved as 'simplified_graph_Aus_no_dis.graphml'")
