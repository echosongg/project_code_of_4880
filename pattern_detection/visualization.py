import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors

# 定义地图的地理范围
domain = [111.975, 156.275, -44.525, -9.975]  # 澳大利亚的大致地理范围

# 加载社区位置数据
community_positions = []
for i in range(1, 7):
    path = f'/home/599/xs5813/4880/pattern_detection/l_result/community_{i}_positions.npy'
    community_positions.append(np.load(path))

# 设置颜色
community_colors = [
    '#FF0000',  # Bright Red
    '#0000FF',  # Bright Blue
    '#00FF00',  # Bright Green
    '#FFFF00',  # Yellow
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#800000',  # Maroon
    '#808000',  # Olive
    '#008080',  # Teal
    '#800080',  # Purple
    '#FF6347',  # Tomato
    '#4682B4',  # SteelBlue
    '#32CD32',  # LimeGreen
    '#FFD700',  # Gold
    '#8A2BE2'   # BlueViolet
]
 # 使用不同颜色表示不同社区

# 初始化地图
fig = plt.figure(figsize=(10, 8))
m = Basemap(projection='merc', llcrnrlon=domain[0], llcrnrlat=domain[2], urcrnrlon=domain[1], urcrnrlat=domain[3], resolution='i')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# 绘制每个社区的位置
for positions, color in zip(community_positions, community_colors):
    if positions.size > 0:  # 检查是否有位置数据
        lat, lon = positions[:, 0], positions[:, 1]
        x, y = m(lon, lat)
        m.scatter(x, y, color=color, label=f'Community {community_colors.index(color) + 1}', edgecolor='k', s=50)

plt.legend()
plt.title('Visualization of Community Positions')

# 保存图像到当前目录下
plt.savefig('/home/599/xs5813/4880/pattern_detection/l_result/Community_Positions_Map.png', dpi=300, bbox_inches='tight')

# 展示图像
plt.show()
