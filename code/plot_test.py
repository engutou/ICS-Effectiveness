import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_efficiency_metrics(attacker_means, attacker_entropies, defender_means, defender_entropies,
                            strategies, title="Mean and entropy of eff. over different strategies", save_path=None):
    """
    绘制攻击者和防御者视角下的效能差均值和熵的对比图

    参数:
    - attacker_means: 攻击者视角下的效能差均值列表
    - attacker_entropies: 攻击者视角下的熵列表
    - defender_means: 防御者视角下的效能差均值列表
    - defender_entropies: 防御者视角下的熵列表
    - strategies: 策略名称列表（x轴标签）
    - title: 图表标题
    - save_path: 保存图表的路径（如果为None，则不保存）
    """
    # 设置中文字体
    # plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建画布和双Y坐标轴
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # 攻击者数据（红色）
    x = np.arange(len(strategies))
    width = 0.35

    # 绘制攻击者效能差均值（左Y轴）
    ax1.plot(x - width / 8, attacker_means, 'o-', color='red', label='mean-attacker', linewidth=2)

    # 绘制攻击者熵（右Y轴）
    ax2.plot(x - width / 8, attacker_entropies, 's--', color='red', label='entropy-attacker', linewidth=2)

    # 防御者数据（蓝色）
    # 绘制防御者效能差均值（左Y轴）
    ax1.plot(x + width / 8, defender_means, 'o-', color='blue', label='mean-defender', linewidth=2)

    # 绘制防御者熵（右Y轴）
    ax2.plot(x + width / 8, defender_entropies, 's--', color='blue', label='entropy-defender', linewidth=2)

    # 设置坐标轴标签和标题
    ax1.set_xlabel('strategy', fontsize=12)
    ax1.set_ylabel('mean', fontsize=12, color='black')
    ax2.set_ylabel('entropy (bits)', fontsize=12, color='black')
    plt.title(title, fontsize=14)

    # 设置x轴刻度和标签
    ax1.set_xticks(x)
    # ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.set_xticklabels(strategies)

    # 添加网格线
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 自动调整布局
    plt.tight_layout()

    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    return plt


# 示例用法
if __name__ == "__main__":
    # 7种攻防策略的数据
    attacker_means = [0.35, 0.42, 0.28, 0.15, 0.22, 0.30, 0.27]  # 攻击者视角下的效能差均值
    attacker_entropies = [1.8, 1.6, 2.0, 2.2, 1.9, 1.7, 1.85]  # 攻击者视角下的熵

    defender_means = [-0.25, -0.32, -0.18, -0.05, -0.12, -0.20, -0.15]  # 防御者视角下的效能差均值
    defender_entropies = [1.9, 1.7, 1.5, 1.3, 1.6, 1.8, 1.7]  # 防御者视角下的熵

    strategies = ["A1/D1", "A2/D2", "A3/D3", "A4/D4", "A5/D5", "A6/D6", "A7/D7"]  # 策略名称

    # 绘制图表
    plt = plot_efficiency_metrics(
        attacker_means, attacker_entropies,
        defender_means, defender_entropies,
        strategies,
        title="mean and entropy of effectiveness over different strategies"
    )

    # 显示图表
    plt.show()