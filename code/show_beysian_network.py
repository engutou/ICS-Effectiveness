import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from matplotlib.patches import FancyArrowPatch

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

# 创建网络结构（转换为networkx标准DiGraph）
G = nx.DiGraph()
G.add_edges_from([
    ('天气', '交通'),
    ('天气', '出勤'),
    ('交通', '出勤')
])

# 使用层次化布局防止节点重叠
pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")  # 需要安装pygraphviz
# 若未安装graphviz，可用以下替代布局：
# pos = nx.planar_layout(G)  # 平面布局
# pos = nx.circular_layout(G, scale=2)  # 圆形布局

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制节点
nx.draw_networkx_nodes(
    G, pos,
    node_size=2000,
    node_color='lightblue',
    edgecolors='black',
    linewidths=2,
    ax=ax
)

# 自定义边绘制方案（使用FancyArrowPatch）
for (u, v) in G.edges():
    # 获取箭头起终点坐标
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    # 创建自定义箭头
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle="arc3,rad=0.2",
        arrowstyle="Simple,head_width=6,head_length=12",
        color="gray",
        linewidth=2,
        mutation_scale=30
    )
    ax.add_patch(arrow)

# 绘制节点标签
nx.draw_networkx_labels(
    G, pos,
    font_size=16,
    font_family='SimHei',
    ax=ax
)

# 设置标题并调整布局
plt.title('天气-交通-出勤贝叶斯网络结构', fontsize=18)
plt.axis('off')
plt.tight_layout()

# 显示图形
plt.show()

# 打印验证信息
print("节点位置验证:")
for node, coords in pos.items():
    print(f"{node}: x={coords[0]:.2f}, y={coords[1]:.2f}")