import matplotlib.pyplot as plt
import numpy as np
from holoviews.plotting.bokeh.styles import font_size


def plot_dict_bars(data_dict, file_path, width=0.3, title="分组柱状图",
                   xlabel="数据组", ylabel="数值",
                   figsize=(12, 7), fontsize=20):
    """
    绘制符合新数据结构的柱状图：
    - 每个数据组（如'data1'）包含两条相邻柱子
    - 第一条柱子高度为第一个二元组的数值（subdata1对应x1）
    - 第二条柱子由后两个二元组数值堆叠而成（subdata2对应y1在下，subdata3对应y2在上）
    - 每组下方xtick标签为组名（如'data1'）
    - 每根柱子顶部标注对应子数据名称（如subdata1、subdata2等）

    参数:
        data_dict: 字典数据，格式为{组名: [('子数据名1', 值1), ('子数据名2', 值2), ('子数据名3', 值3)]}
        width: 单个柱子宽度
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图表大小
        label_fontsize: 柱子顶部标签字体大小
    """
    plt.rcParams["font.size"] = fontsize
    # 解决中文显示问题
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 解析数据
    group_names = list(data_dict.keys())  # 组名列表（data1, data2...）
    n_groups = len(group_names)  # 组数

    # 提取每组的子数据（名称和数值）
    x_names = []  # 第一条柱子的子数据名
    x_values = []  # 第一条柱子的数值
    y_names = []  # 堆叠柱下方子数据名
    y_values = []  # 堆叠柱下方数值
    z_names = []  # 堆叠柱上方子数据名
    z_values = []  # 堆叠柱上方数值

    for group in group_names:
        subdata = data_dict[group]
        x_names.append(subdata[0][0])
        x_values.append(subdata[0][1])
        y_names.append(subdata[1][0])
        y_values.append(subdata[1][1])
        z_names.append(subdata[2][0])
        z_values.append(subdata[2][1])

    # 设置柱子位置（每组包含两个相邻柱子）
    indices = np.arange(n_groups)  # 组索引（0,1,2...）
    x_pos = indices - width / 2  # 第一条柱子（x）的位置
    stacked_pos = indices + width / 2  # 堆叠柱（y+z）的位置

    # 创建图表
    plt.figure(figsize=figsize)

    # 绘制第一条柱子（x值）
    plt.bar(x_pos, x_values, width, label='X类数据', color='#1f77b4')

    # 绘制堆叠柱的下方（y值）
    plt.bar(stacked_pos, y_values, width, label='Y类数据', color='#ff7f0e')

    # 绘制堆叠柱的上方（z值，底部为y值）
    plt.bar(stacked_pos, z_values, width, bottom=y_values, label='Z类数据', color='#2ca02c')

    # 添加柱子顶部的子数据名称标签
    for i in range(n_groups):
        # X柱标签（位置：x_pos[i]，高度：x_values[i]）
        plt.text(x_pos[i], x_values[i], x_names[i],
                 ha='center', va='bottom', fontsize=fontsize)

        # Y柱标签（位置：stacked_pos[i]，高度：y_values[i]）
        plt.text(stacked_pos[i], y_values[i], y_names[i],
                 ha='center', va='bottom', fontsize=fontsize)

        # Z柱标签（位置：stacked_pos[i]，高度：y_values[i]+z_values[i]）
        plt.text(stacked_pos[i], y_values[i] + z_values[i], z_names[i],
                 ha='center', va='bottom', fontsize=fontsize)

    # 设置坐标轴和标题
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.xticks(indices, group_names)  # 每组下方的标签为组名
    # plt.legend()

    # 调整布局并显示
    plt.tight_layout()
    # plt.show()
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    print(f"策略对比图已保存至 {file_path}.png")
    plt.close()  # 关闭图表释放资源


# 使用示例
if __name__ == "__main__":
    import warnings

    # 过滤字体警告
    warnings.filterwarnings(
        "ignore",
        # message=".*findfont.*",  # 仅过滤包含关键字findfont的警告
        category=UserWarning
    )
    # 示例数据：字典格式，键为组名，值为3个二元组（子数据名, 数值）
    sample_data = {
        'data1': [('销量', 150), ('成本', 80), ('利润', 70)],
        'data2': [('销量', 200), ('成本', 110), ('利润', 90)],
        'data3': [('销量', 180), ('成本', 95), ('利润', 85)],
        'data4': [('销量', 220), ('成本', 120), ('利润', 100)]
    }

    # 绘制图表
    plot_dict_bars(
        sample_data,
        file_path=None,
        # title="产品数据对比（销量 vs 成本+利润）",
        title="",
        xlabel="产品类别",
        ylabel="金额（元）",
        width=0.25,  # 调整柱子宽度使标签更清晰
    )