import itertools
import matplotlib.pyplot as plt
import numpy as np


def phi(x):
    # 每个设备的每个状态对效能的贡献
    values = {'0': 1, '1': 0.8, '2': 0.1, '3': 0}
    return values[x]


def f(stats, weights):
    # 每个设备对效能的贡献
    y1 = weights[0] * phi(stats[0])
    y2 = weights[1] * phi(stats[0])
    y3 = weights[2] * phi(stats[0])
    y4 = weights[3] * phi(stats[0])
    return y1 + y2 + y3 + y4


# 定义权重
weights = [0.4, 0.2, 0.2, 0.2]

# 生成所有可能的自变量组合
combinations = list(itertools.product([0, 1, 2, 3], repeat=4))

# 计算每个组合对应的 f 值
f_values = []
decimal_x = []
for comb in combinations:
    x1, x2, x3, x4 = comb
    f_values.append(f(x1, x2, x3, x4, weights))
    binary_str = ''.join([format(i, '02b') for i in comb])
    decimal_x.append(int(binary_str, 2))

# 创建图形
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# 绘制图形，折线用红色加圆点标记
ax.plot(decimal_x, f_values, 'b-o')

# 设置图形属性
ax.set_xlabel('Network States')
ax.set_ylabel('Network Effectiveness')

# 设置横轴范围从 0 到 255
ax.set_xlim(0, 255)

# 只设置 4 的倍数作为 x 轴刻度标签，且不旋转
xticks = [i for i in range(0, 257, 8)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=10.5)

# 打开网格
ax.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
