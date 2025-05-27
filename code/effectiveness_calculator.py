import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


class EffectivenessCalculator:
    def __init__(self, state_weights=None, state_effectiveness=None):
        """
        初始化效能计算参数
        :param state_weights: 各设备权重列表，顺序对应X1-X4
        :param state_effectiveness: 各状态的效能贡献字典，格式为{状态值: 贡献值}
        """
        # 默认参数采用论文案例中的定义
        self.weights = state_weights if state_weights else [0.4, 0.2, 0.2, 0.2]
        self.effectiveness = state_effectiveness if state_effectiveness else {'0': 1, '1': 0.8, '2': 0.1, '3': 0}
        self.num_devices = len(self.weights)
        self.states = list(self.effectiveness.keys())

    def calculate_efficiency(self, state_vector):
        """
        计算单个状态向量的效能值
        :param state_vector: 设备状态向量，如(0,1,0,3)
        :return: 效能值
        """
        if len(state_vector) != self.num_devices:
            raise ValueError("状态向量长度必须与设备数量一致")
        return sum(self.weights[i] * self.effectiveness[state]
                   for i, state in enumerate(state_vector))

    # def generate_state_space(self):
    #     """生成所有可能的状态组合（离散状态空间）"""
    #     return list(product(self.states, repeat=self.num_devices))

    def compute_efficiency_distribution(self, joint_distribution):
        """
        计算效能分布
        :param joint_distribution: X(t)和X(t+1)的联合分布字典，格式为{(x_t, x_t1): probability}
        :return: 效能差分布字典 {delta_f: probability}
        """
        # state_space = self.generate_state_space()
        eff_dist = {}

        for (x_t, x_t1), prob in joint_distribution.items():
            eff_t = self.calculate_efficiency(x_t)
            eff_t1 = self.calculate_efficiency(x_t1)
            delta_f = eff_t1 - eff_t

            if delta_f in eff_dist:
                eff_dist[delta_f] += prob
            else:
                eff_dist[delta_f] = prob

        # 标准化概率分布（处理浮点精度误差）
        total = sum(eff_dist.values())
        if abs(total - 1.0) > 1e-9:
            eff_dist = {k: v / total for k, v in eff_dist.items()}

        return eff_dist

    def calculate_statistics(self, eff_dist):
        """
        计算效能均值和信息熵
        :param eff_dist: 效能差分布字典 {delta_f: probability}
        :return: (均值, 信息熵)
        """
        delta_f, probs = zip(*eff_dist.items())
        mean = np.sum([df * p for df, p in zip(delta_f, probs)])
        entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])  # 使用log2计算信息熵
        return mean, entropy


def plot_efficiency_distribution(eff_dist, title="Effectiveness Difference Distribution", show_mean=True):
    """
    绘制效能差分布折线图

    参数:
    - eff_dist: 效能差分布字典 {delta_f: probability}
    - title: 图表标题
    - show_mean: 是否显示均值线
    """
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 准备数据
    delta_f, probs = zip(*sorted(eff_dist.items()))

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制折线图（使用marker显示离散点）
    plt.plot(delta_f, probs, '-', color='skyblue', linewidth=2)

    # # 添加数据标签
    # for x, y in zip([f"{df:.2f}" for df in delta_f], probs):
    #     plt.annotate(f'{y:.2%}',
    #                  (x, y),
    #                  textcoords="offset points",
    #                  xytext=(0, 10),
    #                  ha='center',
    #                  fontsize=9)

    # 添加均值线
    if show_mean:
        mean = np.sum([df * p for df, p in zip(delta_f, probs)])
        plt.axvline(x=mean, color='r', linestyle='--', label=f'Mean: {mean:.4f}')

        plt.legend()

    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.xlabel('ΔF', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.xticks(rotation=45)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return plt


# 示例用法
if __name__ == "__main__":
    # 示例联合分布：假设X(t)和X(t+1)的联合概率分布（需根据实际输入调整）
    # 格式：{(x_t_state_vector, x_t1_state_vector): probability}
    # 状态向量顺序：(X1, X2, X3, X4)，每个元素为0-3的整数
    sample_joint_dist = {
        (('0', '0', '0', '0'), ('0', '0', '0', '0')): 0.2,
        (('0', '0', '0', '0'), ('1', '0', '0', '0')): 0.1,
        (('0', '0', '0', '0'), ('0', '1', '0', '0')): 0.1,
        (('0', '0', '0', '0'), ('0', '0', '1', '0')): 0.1,
        (('0', '0', '0', '0'), ('0', '0', '0', '1')): 0.1,
        (('0', '0', '0', '0'), ('2', '0', '0', '0')): 0.1,
        (('0', '0', '0', '0'), ('0', '2', '0', '0')): 0.1,
        (('0', '0', '0', '0'), ('0', '0', '2', '0')): 0.1,
        (('0', '0', '0', '0'), ('0', '0', '0', '2')): 0.1
    }
    # 初始化计算实例
    calculator = EffectivenessCalculator()

    # 计算效能分布
    eff_dist = calculator.compute_efficiency_distribution(sample_joint_dist)
    print("效能差分布:（负效能差表示网络效能下降）")
    for delta_f, prob in sorted(eff_dist.items(), key=lambda x: x[0]):
        print(f"ΔF={delta_f:.2f}: P={prob:.4f}")

    # 计算统计量
    mean, entropy = calculator.calculate_statistics(eff_dist)
    print(f"\n效能均值: {mean:.4f}")
    print(f"信息熵: {entropy:.4f} bit")

    # 绘制效能分布图表
    plt = plot_efficiency_distribution(eff_dist)
    plt.savefig('eff_dist.png', dpi=300, bbox_inches='tight')
    plt.show()
