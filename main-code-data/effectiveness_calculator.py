import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import pickle
import os.path

from pgmpy.factors.discrete import DiscreteFactor


class EffectivenessCalculator:
    def __init__(
        self,
        var_order: List[str],  # 变量顺序（需与联合分布一致，前半为t时刻，后半为t+1时刻）
        node_weights: Dict[str, float],  # 变量权重字典 {变量名: 权重值}
        state_effectiveness: Dict[str, Dict[int, float]],  # 状态效能字典 {变量名: {状态索引: 效能值}}
        cache_file_path: str = None  # 新增：缓存文件路径参数
    ):
        """
        初始化效能计算器，支持不同变量的不同状态数，通过np.ndindex高效生成状态索引
        """
        # 存储核心参数
        self.var_order = var_order
        self.num_vars = len(var_order)
        self.node_weights = node_weights
        self.state_effectiveness = state_effectiveness
        self.cache_file_path = cache_file_path

        # 1. 校验变量完整性（权重和效能必须包含所有变量）
        self._validate_variable_completeness()

        # 2. 动态提取每个变量的状态数并校验状态连续性（必须从0开始连续）
        self.var_state_count = {}  # {变量名: 状态数}
        self.var_valid_states = {}  # {变量名: 有效状态列表}
        self._extract_and_validate_states()

        # 3. 拆分t时刻和t+1时刻变量（假设变量总数为偶数，前后半部分数量相等）
        self.num_t_vars = self.num_vars // 2
        self.num_t1_vars = self.num_vars - self.num_t_vars
        self._validate_var_grouping()  # 校验分组合理性
        self.var_order_t = var_order[:self.num_t_vars]  # t时刻变量顺序
        self.var_order_t1 = var_order[self.num_t_vars:]  # t+1时刻变量顺序

        # 4. 计算索引权重（用于状态→缓存索引的转换）
        self.index_weights = self._compute_index_weights()

        # 5. 处理缓存逻辑：如果文件存在则加载，否则预计算并保存
        if self.cache_file_path and os.path.exists(self.cache_file_path):
            print(f"[效能缓存] 加载已存在的缓存文件: {self.cache_file_path}")
            with open(self.cache_file_path, 'rb') as f:
                self.delta_e_cache = pickle.load(f)
        else:
            # 预计算所有状态组合的效能差缓存
            self.delta_e_cache = self._precompute_delta_e_cache()
            # 如果指定了缓存路径，保存到文件
            if self.cache_file_path:
                print(f"[效能缓存] 保存缓存到文件: {self.cache_file_path}")
                with open(self.cache_file_path, 'wb') as f:
                    pickle.dump(self.delta_e_cache, f)

    def _validate_variable_completeness(self) -> None:
        """校验权重和效能字典包含所有变量"""
        for var in self.var_order:
            if var not in self.node_weights:
                raise KeyError(f"节点权重缺失变量：{var}（变量顺序：{self.var_order}）")
            if var not in self.state_effectiveness:
                raise KeyError(f"状态效能缺失变量：{var}（变量顺序：{self.var_order}）")

    def _extract_and_validate_states(self) -> None:
        """提取每个变量的状态数，并校验状态是否从0开始连续"""
        for var in self.var_order:
            # 提取并排序状态值（确保顺序一致）
            states = sorted(self.state_effectiveness[var].keys())
            # 校验状态是否为非负整数且连续（如[0,1,2]有效，[0,2]或[1,2]无效）
            if not all(isinstance(s, int) and s >= 0 for s in states):
                raise ValueError(f"变量{var}的状态必须为非负整数，当前状态：{states}")
            if states != list(range(len(states))):
                raise ValueError(f"变量{var}的状态不连续（需从0开始），当前状态：{states}")
            self.var_valid_states[var] = states
            self.var_state_count[var] = len(states)

    def _validate_var_grouping(self) -> None:
        """校验t和t+1时刻变量数量是否相等"""
        if self.num_t_vars != self.num_t1_vars:
            raise ValueError(
                f"变量总数必须为偶数（t和t+1时刻变量数量需相等），"
                f"当前总数：{self.num_vars}，t时刻：{self.num_t_vars}，t+1时刻：{self.num_t1_vars}"
            )

    def _compute_index_weights(self) -> List[int]:
        """
        按变量顺序反向计算权重（最后一个变量权重为1，前面的变量权重 = 后续所有变量状态数的乘积）
        匹配pgmpy的笛卡尔积顺序（最后一个变量变化最快）
        例：变量0（4种状态）→ 权重 2x3=6；变量1（3种状态）→ 权重 2；变量2（2种状态）→ 权重 1
        """
        weights = [1] * self.num_vars  # 第0个变量权重默认为1
        # 从倒数第二个变量开始向前计算（最后一个变量权重固定为1）
        for i in range(self.num_vars - 2, -1, -1):
            # 当前变量的权重 = 下一个变量的权重 × 下一个变量的状态数
            next_var = self.var_order[i + 1]
            weights[i] = weights[i + 1] * self.var_state_count[next_var]
        return weights

    def _precompute_delta_e_cache(self) -> np.ndarray:
        """
        预计算所有状态组合的效能差，使用np.ndindex高效生成状态索引
        缓存格式：一维数组，索引为状态组合的唯一编码，值为效能差（E_t1 - E_t）
        """
        # 生成var_dims（每个变量的状态数，按var_order顺序）
        var_dims = [self.var_state_count[var] for var in self.var_order]
        # 计算总状态组合数（所有变量状态数的乘积）
        total_combinations = 1
        for dim in var_dims:
            total_combinations *= dim
        print(f"[效能缓存] 总状态组合数：{total_combinations}（开始预计算...）")

        # 初始化缓存数组（用float32节省内存）
        delta_e_cache = np.zeros(total_combinations, dtype=np.float32)

        cnt = 0  # 计数器：已处理的状态数
        progress_step = 500000  # 每处理50万输出一次进度
        # 用np.ndindex生成所有状态组合的索引（同时作为状态值元组）
        for states in np.ndindex(*var_dims):
            # states是元组，如(0,1,2,0,1)，按var_order顺序对应每个变量的状态
            # 计算该状态组合在缓存中的索引
            cache_index = self._states_to_index(states)
            # 拆分t和t+1时刻的状态
            states_t = states[:self.num_t_vars]
            states_t1 = states[self.num_t_vars:]
            # 计算t时刻效能（E_t）
            e_t = sum(
                self.node_weights[var] * self.state_effectiveness[var][s]
                for var, s in zip(self.var_order_t, states_t)
            )
            # 计算t+1时刻效能（E_t1）
            e_t1 = sum(
                self.node_weights[var] * self.state_effectiveness[var][s]
                for var, s in zip(self.var_order_t1, states_t1)
            )
            # 存储效能差（E_t1 - E_t）
            delta_e_cache[cache_index] = e_t1 - e_t

            # 进度显示逻辑
            cnt += 1
            # 每100万输出一次进度（或处理完最后一个时输出）
            if cnt % progress_step == 0 or cnt == total_combinations:
                progress = cnt / total_combinations * 100  # 进度百分比
                print(f"[效能缓存] 已处理 {cnt:,} 个状态（{progress:.2f}%）")

        print(f"[效能缓存] 预计算完成，缓存大小：{delta_e_cache.nbytes / 1024 / 1024:.2f} MB")
        return delta_e_cache

    def _states_to_index(self, states: Tuple[int, ...]) -> int:
        """
        将状态元组（按var_order顺序）转换为缓存的唯一索引
        :param states: 状态元组，如(0,1,2,0,1)
        :return: 缓存索引（整数）
        """
        if len(states) != self.num_vars:
            raise ValueError(
                f"状态元组长度必须为{self.num_vars}（与变量数一致），"
                f"当前长度：{len(states)}"
            )
        # 校验每个状态是否在变量的有效范围内
        for i, (var, s) in enumerate(zip(self.var_order, states)):
            if s not in self.var_valid_states[var]:
                raise ValueError(
                    f"变量{var}的状态{s}无效，有效状态：{self.var_valid_states[var]}"
                )
        # 计算索引（基于预计算的权重）
        return sum(s * self.index_weights[i] for i, s in enumerate(states))

    def get_delta_e(self, states: Tuple[int, ...]) -> float:
        """
        通过状态元组获取效能差（查表操作）
        :param states: 状态元组（按var_order顺序）
        :return: 效能差（E_t1 - E_t）
        """
        cache_index = self._states_to_index(states)
        return self.delta_e_cache[cache_index]

    def compute_efficiency_distribution_from_pgm(
            self,
            pgm_infer_result: DiscreteFactor
    ) -> Dict[float, float]:
        """
        从pgmpy的推理结果（Factor对象）计算效能差分布（优化版：向量化操作提升效率）
        """
        # 1. 校验变量顺序一致性（同原逻辑）
        infer_vars = pgm_infer_result.variables
        if infer_vars != self.var_order:
            raise ValueError(f"推理变量与计算器变量不匹配！\n推理: {infer_vars}\n计算器: {self.var_order}")

        # 2. 提取概率数组并与效能差缓存配对
        probabilities = pgm_infer_result.values.flatten().astype(np.float32)  # 概率序列
        delta_e_array = self.delta_e_cache.astype(np.float32)  # 效能差序列（与概率序列顺序一致）

        # 3. 过滤极小概率（减少无效计算）
        valid_mask = probabilities > 1e-12  # 有效概率的掩码
        valid_probs = probabilities[valid_mask]  # 有效概率
        valid_deltas = delta_e_array[valid_mask]  # 对应的有效效能差

        # 4. 合并相同效能差的概率（核心优化）
        # 4.1 获取唯一效能差及其索引
        unique_deltas, indices = np.unique(valid_deltas, return_inverse=True)
        # 4.2 按索引分组求和概率
        merged_probs = np.bincount(indices, weights=valid_probs)

        # 5. 处理浮点误差（标准化概率和为1）
        total_prob = merged_probs.sum()
        if not np.isclose(total_prob, 1.0, atol=1e-9):
            merged_probs /= total_prob
            print(f"[警告] 概率总和为{total_prob:.6f}，已标准化为1.0")

        # 转换为字典返回
        return dict(zip(unique_deltas, merged_probs))

    def compute_efficiency_distribution_from_pgm2(
            self,
            pgm_infer_result: DiscreteFactor
    ) -> Dict[float, float]:
        """
        note: 该方法很慢，没优化，不用了
        从pgmpy的推理结果（Factor对象）直接计算效能差分布
        :param pgm_infer_result: pgmpy的VariableElimination推理结果（需包含所有t和t+1时刻变量）
        :return: 效能差分布字典 {效能差: 总概率}
        """
        # 1. 校验推理结果变量与计算器变量顺序的一致性
        infer_vars = pgm_infer_result.variables
        # 确保变量顺序严格一致（状态元组顺序依赖于此）
        if infer_vars != self.var_order:
            raise ValueError(
                f"推理结果变量与计算器变量不匹配！\n"
                f"推理变量: {infer_vars}\n"
                f"计算器变量: {self.var_order}"
            )

        # 2. 提取状态组合与概率（从Factor对象中）
        # 生成所有状态组合的索引（pgmpy的values是按顺序排列的概率数组）
        state_indices = np.ndindex(*pgm_infer_result.cardinality)
        probabilities = pgm_infer_result.values.flatten()  # 概率数组（与状态组合顺序对应）

        # 计算总状态组合数（所有变量状态数的乘积）
        total_combinations = 1
        for dim in pgm_infer_result.cardinality:
            total_combinations *= dim
        print(f"[效能差分布计算] 总状态组合数：{total_combinations}（开始计算...）")

        eff_dist = {}
        cnt = 0  # 计数器：已处理的状态数
        progress_step = 500000  # 每处理50万输出一次进度
        for idx, prob in zip(state_indices, probabilities):
            # 进度显示逻辑
            cnt += 1
            # 每50万输出一次进度（或处理完最后一个时输出）
            if cnt % progress_step == 0 or cnt == total_combinations:
                progress = cnt / total_combinations * 100  # 进度百分比
                print(f"[效能差分布计算] 已处理 {cnt:,} 个状态（{progress:.2f}%）")
            if prob <= 1e-12:  # 过滤极小概率（避免浮点误差）
                continue
            # note: self.state_effectiveness中存的就是 索引->效能
            # idx是元组，每个元素是变量在其状态列表中的索引（如0→第一个状态）
            # 3. 计算效能差并累计概率
            delta_e = self.get_delta_e(idx)
            eff_dist[delta_e] = eff_dist.get(delta_e, 0.0) + prob

        # 4. 标准化概率（处理浮点误差）
        total_prob = sum(eff_dist.values())
        if abs(total_prob - 1.0) > 1e-9:
            print(f"[警告] 概率总和为{total_prob:.6f}，已标准化为1.0")
            eff_dist = {k: v / total_prob for k, v in eff_dist.items()}

        return eff_dist


def calculate_statistics(eff_distribution):
    """
    计算效能均值和信息熵
    :param eff_distribution: 效能差分布字典 {delta_f: probability}
    :return: (均值, 信息熵)
    """
    delta_eff, probs = zip(*eff_distribution.items())
    mean_eff = np.sum([df * p for df, p in zip(delta_eff, probs)])
    entropy_eff = -np.sum([p * np.log2(p) for p in probs if p > 0])  # 使用log2计算信息熵
    return mean_eff, entropy_eff


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
    plt.plot(delta_f, probs, '-o', color='skyblue', linewidth=2)

    # 添加均值线
    if show_mean:
        m = np.sum([df * p for df, p in zip(delta_f, probs)])  # 求均值
        plt.axvline(x=m, color='r', linestyle='--', label=f'Mean: {m:.4f}')

        plt.legend()

    # 设置图表属性
    plt.title(title, fontsize=14)
    plt.xlabel('ΔF', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return plt


# 示例用法
if __name__ == "__main__":
    num_nodes = 7
    network_state_names = [f'X_{i}' for i in range(1, num_nodes+1)]
    network_state_names_t1 = [f'X_{i}_t1' for i in range(1, num_nodes+1)]
    node_weights_list = [0.2, 0.12, 0.3, 0.16, 0.12, 0.05, 0.05]
    assert num_nodes == len(node_weights_list)

    ordered_vars = network_state_names + network_state_names_t1
    var_weights = {network_state_names[i]:node_weights_list[i] for i in range(0, num_nodes)}
    var_weights_t1 = {network_state_names_t1[i]: node_weights_list[i] for i in range(0, num_nodes)}
    var_weights = var_weights | var_weights_t1

    # 每种状态对应的效能，索引->效能
    state_idx2eff = {var: {0: 1.0, 1: 0.8, 2: 0.3, 3: 0.0} for var in ordered_vars}

    # 初始化计算器
    calculator = EffectivenessCalculator(
        var_order=ordered_vars,
        node_weights=var_weights,
        state_effectiveness=state_idx2eff,
        cache_file_path='delta_e_reverse.pkl'
    )
