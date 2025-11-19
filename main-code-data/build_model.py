import json
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
from itertools import product
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD


class MyBayesianNetwork:
    """
    # 我的贝叶斯网络类
    # 主要功能:
       1、从json文件中读取贝叶斯网络的结构信息
       2、初始化DiscreteBayesianNetwork
       3、根据贝叶斯网络的结构信息为每个节点生成csv格式的CPD文件
    """

    def __init__(self):
        self.variables = {}  # 变量名 -> 取值列表
        self.parents = defaultdict(list)  # 变量 -> 父节点列表
        self.children = defaultdict(list)  # 变量 -> 子节点列表
        self.model = None
        self.data_dir = None  # 新增：存储json文件和cpd文件的目录

    def add_variable(self, name, values):
        """添加变量"""
        if name in self.variables:
            raise ValueError(f"变量 '{name}' 已存在")

        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"变量 '{name}' 必须至少有两个可能的取值")

        self.variables[name] = values

    def add_edge(self, from_var, to_var):
        """添加边"""
        if from_var not in self.variables:
            raise ValueError(f"变量 '{from_var}' 不存在")
        if to_var not in self.variables:
            raise ValueError(f"变量 '{to_var}' 不存在")

        # 检查是否已存在
        if to_var not in self.children[from_var]:
            self.children[from_var].append(to_var)

        if from_var not in self.parents[to_var]:
            self.parents[to_var].append(from_var)

    def get_variable_values(self, variable_name):
        """获取变量的可能取值"""
        if variable_name not in self.variables:
            raise ValueError(f"变量 '{variable_name}' 不存在")
        return self.variables[variable_name]

    def get_parents(self, variable_name):
        """获取变量的父节点"""
        if variable_name not in self.variables:
            raise ValueError(f"变量 '{variable_name}' 不存在")
        return self.parents[variable_name].copy()

    def get_children(self, variable_name):
        """获取变量的子节点"""
        if variable_name not in self.variables:
            raise ValueError(f"变量 '{variable_name}' 不存在")
        return self.children[variable_name].copy()

    def get_all_variables(self):
        """获取所有变量名称"""
        return list(self.variables.keys())

    def is_valid_network(self):
        """验证网络是否为有效的有向无环图"""
        visited = set()
        recursion_stack = set()

        def has_cycle(node):
            """检查是否有环"""
            if node in recursion_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            recursion_stack.add(node)

            for child in self.children.get(node, []):
                if has_cycle(child):
                    return True

            recursion_stack.remove(node)
            return False

        for node in self.variables:
            if has_cycle(node):
                return False
        return True

    def set_model(self):
        """
        根据贝叶斯网络的结构初始化 DiscreteBayesianNetwork 模型
        :return:
        """
        # 将网络中的边（父节点与子节点的关系）转换为(from_var, to_var)格式的元组列表
        bn_edges = []
        for to_var, from_vars in self.parents.items():
            for from_var in from_vars:
                bn_edges.append((from_var, to_var))
        # 模型初始化
        self.model = DiscreteBayesianNetwork(bn_edges)

    def _generate_valid_probabilities(self, target_card: int, parent_combination: tuple = None,
                                      parents: list = None) -> list:
        """
        生成符合逻辑的概率分布（确保和为1，且贴合工控场景）
        note:只是生成csv格式的CPD文件模板时使用，真的贝叶斯网络推理时不用它产生的数据
        :param target_card: 目标变量的取值数量
        :param parent_combination: 当前父节点取值组合（用于子节点概率依赖）
        :param parents: 父节点列表（用于获取父节点取值含义）
        :return: 归一化后的概率列表
        """
        # 根节点（无父节点）：默认第一个取值为“正常”，概率占比最高
        if parent_combination is None:
            # 正常状态（第一个取值）概率 0.8-0.9，其余取值分配剩余概率
            base_probs = np.random.uniform(0.8, 0.9, size=1)
            remaining_probs = np.random.dirichlet(np.ones(target_card - 1)) * (1 - base_probs[0])
            probs = np.concatenate([base_probs, remaining_probs])
        else:
            # 子节点：父节点状态越“异常”（取值越大），子节点异常概率越高
            # 假设父节点取值为字符串类型的数字（如"0"=正常，"3"=完全失效）
            parent_abnormal_level = 0
            for p_val in parent_combination:
                try:
                    # 累加所有父节点的异常等级（取值越大越异常）
                    parent_abnormal_level += int(p_val)
                except (ValueError, TypeError):
                    # 非数字取值按1计算（默认轻度异常）
                    parent_abnormal_level += 1

            # 异常等级越高，子节点正常状态概率越低
            normal_prob = max(0.1, 0.9 - parent_abnormal_level * 0.15)  # 最低保留10%正常概率
            base_probs = np.array([normal_prob])
            remaining_probs = np.random.dirichlet(np.ones(target_card - 1)) * (1 - normal_prob)
            probs = np.concatenate([base_probs, remaining_probs])

        # 确保概率和严格等于1（解决浮点误差）
        probs = probs / probs.sum()
        probs = probs.round(4)
        total = probs.sum()
        if not np.isclose(total, 1.0):
            # 找到最接近0.5的概率值进行调整（减少对分布的影响）
            mid_idx = np.abs(probs - 0.5).argmin()
            probs[mid_idx] += (1.0 - total)
        return probs.round(4).tolist()

    def generate_all_cpd_csv(self):
        """
        为网络中所有节点自动生成CPD CSV文件，文件名格式：var_name_cpd.csv
        """
        if not self.variables:
            raise RuntimeError("请先加载贝叶斯网络结构（如调用from_json_file方法）")

        if self.data_dir is None:
            self.data_dir = os.getcwd()  # 默认使用当前工作目录
            print(f"警告：未指定数据目录，将使用当前工作目录：{self.data_dir}")
        os.makedirs(self.data_dir, exist_ok=True)  # 确保目录存在

        # 遍历所有变量，逐个生成CPD
        for var_name in self.get_all_variables():
            # 1. 获取目标变量的核心信息
            target_values = self.get_variable_values(var_name)  # 目标变量取值（如["0","1","2","3"]）
            parents = self.get_parents(var_name)  # 父节点列表
            target_card = len(target_values)  # 目标变量基数，也就是目标变量的状态个数

            # 2. 构建CSV表头字段
            # 基础字段：target_variable, parents, target_values
            header = ["target_variable", "parents", "target_values"]

            # 父节点列字段（格式：父节点名(parent_序号)）
            if parents:
                parent_cols = [f"{p}(parent_{i + 1})" for i, p in enumerate(parents)]
                header.extend(parent_cols)
            else:
                # 无父节点时，父节点列为parent_placeholder
                parent_cols = ["parent_placeholder"]
                header.extend(parent_cols)

            # 概率列字段（格式：var_name=取值）
            prob_cols = [f"{var_name}={val}" for val in target_values]
            header.extend(prob_cols)

            # 3. 生成CSV行数据
            rows = []
            # 生成父节点所有可能的取值组合（笛卡尔积）
            if parents:
                parent_value_lists = [self.get_variable_values(p) for p in parents]
                parent_combinations = list(product(*parent_value_lists))
            else:
                # 无父节点时，仅一行数据（先验分布）
                parent_combinations = [("None",)]

            # 为每个父节点组合生成一行CPD数据
            for combo in parent_combinations:
                # 初始化行数据，并填充基础字段
                row = [var_name,
                       ",".join(parents) if parents else "None",
                       ",".join(target_values)]
                # 填充父节点取值
                row.extend(combo)
                # 生成并填充概率
                probs = self._generate_valid_probabilities(
                    target_card=target_card,
                    parent_combination=combo if parents else None,
                    parents=parents
                )
                row.extend(probs)

                rows.append(row)

            # 4. 构建DataFrame并写入CSV
            df = pd.DataFrame(rows, columns=header)
            csv_filename = f"{var_name}_cpd.csv"
            csv_path = os.path.join(self.data_dir, csv_filename)  # 使用data_dir路径
            df.to_csv(csv_path, sep=";", index=False, encoding="utf-8")
            print(f"成功生成CPD文件：{csv_path}")

    def read_part_cpd_from_csv(self, csv_filename):
        """
        从data_dir目录读取CPD文件,但是不添加到模型
        格式：target_variable;parents;target_values;父节点列(无空格);概率列
        """
        if self.data_dir is not None:
            csv_path = os.path.join(self.data_dir, csv_filename)
        else:
            csv_path = csv_filename  # 未指定目录时使用传入的路径

        try:
            # 读取CSV，指定分号分隔
            df = pd.read_csv(csv_path, sep=';', comment='#')
            if df.empty:
                raise ValueError(f"CPD文件 {csv_path} 为空")

            # print(df.shape)
            # print(df)

            # 获取目标变量名，并验证变量是否存在于网络中
            target_var = df['target_variable'].iloc[0]
            if target_var not in self.variables:
                raise ValueError(f"目标变量 {target_var} 不在网络中")

            # 获取目标变量取值，并验证是否和json文件中的定义一致
            target_values = [str(v).strip() for v in df['target_values'].iloc[0].split(',')]
            try:
                target_values_json = self.get_variable_values(target_var)
            except ValueError:
                raise ValueError(f"变量 {target_var} 未在JSON结构文件中定义")
            # note:只验证内容，不验证顺序
            if sorted(target_values) != sorted(target_values_json):
                raise ValueError(
                    f"变量 {target_var} 的取值在CSV与JSON中不一致！\n"
                    f"CSV定义: {target_values}\n"
                    f"JSON定义: {target_values_json}"
                )

            # 提取概率列并验证命名与取值的对应关系
            prob_cols = [col.strip() for col in df.columns if f'{target_var}=' in col]
            # 从概率列名中提取取值（如从"X_3=0"中提取"0"）
            prob_col_values = [col.split('=')[1].strip() for col in prob_cols]
            # 概率列对应的取值必须与目标变量取值完全一致（包括顺序）
            if prob_col_values != target_values:
                raise ValueError(
                    f"变量 {target_var} 的概率列命名与目标取值不匹配！\n"
                    f"概率列对应的取值: {prob_col_values}\n"
                    f"目标变量的取值: {target_values}"
                )
            # 验证每组父节点取值对应的概率和为1
            prob_matrix = df[prob_cols].values  # 原始概率矩阵（未转置）
            row_sums = prob_matrix.sum(axis=1)  # 计算每行的概率和（每组父节点组合的概率和）
            if not np.allclose(row_sums, 1.0, atol=1e-6):  # 允许±1e-6的浮点误差
                # 找出不符合的行
                invalid_rows = [i for i, sum_val in enumerate(row_sums) if not np.isclose(sum_val, 1.0, atol=1e-6)]
                raise ValueError(
                    f"变量 {target_var} 的CPD概率和不为1！\n"
                    f"异常行索引: {invalid_rows}\n"
                    f"对应概率和: {[row_sums[i] for i in invalid_rows]}"
                )

            # 提取父节点信息列
            parents_raw = df['parents'].iloc[0]  # 可能为字符串或NaN
            # 父节点为空的情况（NaN或空字符串）
            if pd.isna(parents_raw) or str(parents_raw).strip().lower() in ['', 'none', 'nan']:
                parents = []  # 无父节点时为空列表
            else:
                parents = [p.strip() for p in str(parents_raw).split(',')]  # 移除空格

            # 验证父节点是否存在于网络中
            for p in parents:
                if p not in self.variables:
                    raise ValueError(f"父节点 {p} 不在网络中")

            # 验证文件中的父节点是网络结构中定义的父节点的子集
            # note: 和读取完整CPD的唯一差别
            actual_parents = self.get_parents(target_var)
            if not set(parents).issubset(set(actual_parents)):
                raise ValueError(
                    f"CPD父节点和网络结构不符：{set(parents).difference(actual_parents)}未定义"
                )

            # 提取父节点取值列（匹配格式：父节点名(parent_序号)，无空格）
            parent_cols = [col for col in df.columns if '(parent_' in col]
            # 验证父节点取值列与parents字段的顺序一致性
            expected_parent_cols = [f"{p}(parent_{i + 1})" for i, p in enumerate(parents)]
            if parent_cols != expected_parent_cols:
                raise ValueError(
                    f"父节点列顺序与parents字段不符！\n"
                    f"预期列（按parents顺序）: {expected_parent_cols}\n"
                    f"实际列: {parent_cols}"
                )

            # 生成父节点取值的标准顺序（笛卡尔积）
            parent_value_lists = [self.get_variable_values(p) for p in parents]  # 每个父节点的取值列表
            standard_combinations = list(product(*parent_value_lists))  # 所有组合的标准顺序
            # num_standard_combinations = len(standard_combinations)
            # 解析CSV中的父节点组合
            csv_combinations = []
            for _, row in df.iterrows():
                # 提取每行的父节点取值（按parent_cols顺序）
                combo = tuple(str(row[col]).strip() for col in parent_cols)
                csv_combinations.append(combo)
            # 验证父节点取值的顺序符合标准顺序
            if standard_combinations != csv_combinations:
                raise ValueError(
                    f"父节点取值顺序与标准笛卡尔积的顺序不符！\n"
                    f"标准顺序: {standard_combinations}\n"
                    f"实际顺序: {csv_combinations}"
                )

            # 解析父节点基数（每个父节点的取值数量）
            parent_cardinalities = [len(self.get_variable_values(p)) for p in parents]
            # 目标变量基数
            target_cardinality = len(target_values)
            # 提取概率矩阵并转置（适配pgmpy的TabularCPD格式）
            # 转置后：每行对应目标变量的一个取值，每列对应一组父节点取值组合
            prob_matrix = df[prob_cols].values.T

            # 创建TabularCPD对象
            state_names = {target_var: target_values}
            for parent in parents:
                state_names[parent] = self.get_variable_values(parent)
            cpd = TabularCPD(
                variable=target_var,
                variable_card=target_cardinality,
                values=prob_matrix,
                evidence=parents,
                evidence_card=parent_cardinalities,
                state_names=state_names
            )

            print(f"成功为变量 {target_var} 读取部分CPD（文件：{csv_path}）")
            return cpd

        except FileNotFoundError:
            raise FileNotFoundError(f"CPD文件 {csv_path} 不存在")
        except pd.errors.ParserError:
            raise ValueError(f"文件 {csv_path} 格式错误，请检查分号分隔是否正确")
        except IndexError:
            raise ValueError(f"文件 {csv_path} 缺少必要字段（如target_variable、parents等）")
        except Exception as e:
            raise RuntimeError(f"处理CPD时发生错误：{str(e)}")

    def add_cpd_from_csv(self, csv_filename):
        """
        从data_dir目录读取CPD文件并添加到模型
        格式：target_variable;parents;target_values;父节点列(无空格);概率列
        """
        if self.model is None:
            raise RuntimeError("请先调用set_model()初始化模型")

        if self.data_dir is not None:
            csv_path = os.path.join(self.data_dir, csv_filename)
        else:
            csv_path = csv_filename  # 未指定目录时使用传入的路径

        try:
            # 读取CSV，指定分号分隔
            df = pd.read_csv(csv_path, sep=';', comment='#')
            if df.empty:
                raise ValueError(f"CPD文件 {csv_path} 为空")

            # print(df.shape)
            # print(df)

            # 获取目标变量名，并验证变量是否存在于网络中
            target_var = df['target_variable'].iloc[0]
            if target_var not in self.variables:
                raise ValueError(f"目标变量 {target_var} 不在网络中")

            # 获取目标变量取值，并验证是否和json文件中的定义一致
            target_values = [str(v).strip() for v in df['target_values'].iloc[0].split(',')]
            try:
                target_values_json = self.get_variable_values(target_var)
            except ValueError:
                raise ValueError(f"变量 {target_var} 未在JSON结构文件中定义")
            # note:只验证内容，不验证顺序
            if sorted(target_values) != sorted(target_values_json):
                raise ValueError(
                    f"变量 {target_var} 的取值在CSV与JSON中不一致！\n"
                    f"CSV定义: {target_values}\n"
                    f"JSON定义: {target_values_json}"
                )

            # 提取概率列并验证命名与取值的对应关系
            prob_cols = [col.strip() for col in df.columns if f'{target_var}=' in col]
            # 从概率列名中提取取值（如从"X_3=0"中提取"0"）
            prob_col_values = [col.split('=')[1].strip() for col in prob_cols]
            # 概率列对应的取值必须与目标变量取值完全一致（包括顺序）
            if prob_col_values != target_values:
                raise ValueError(
                    f"变量 {target_var} 的概率列命名与目标取值不匹配！\n"
                    f"概率列对应的取值: {prob_col_values}\n"
                    f"目标变量的取值: {target_values}"
                )
            # 验证每组父节点取值对应的概率和为1
            prob_matrix = df[prob_cols].values  # 原始概率矩阵（未转置）
            row_sums = prob_matrix.sum(axis=1)  # 计算每行的概率和（每组父节点组合的概率和）
            if not np.allclose(row_sums, 1.0, atol=1e-6):  # 允许±1e-6的浮点误差
                # 找出不符合的行
                invalid_rows = [i for i, sum_val in enumerate(row_sums) if not np.isclose(sum_val, 1.0, atol=1e-6)]
                raise ValueError(
                    f"变量 {target_var} 的CPD概率和不为1！\n"
                    f"异常行索引: {invalid_rows}\n"
                    f"对应概率和: {[row_sums[i] for i in invalid_rows]}"
                )

            # 提取父节点信息列
            parents_raw = df['parents'].iloc[0]  # 可能为字符串或NaN
            # 父节点为空的情况（NaN或空字符串）
            if pd.isna(parents_raw) or str(parents_raw).strip().lower() in ['', 'none', 'nan']:
                parents = []  # 无父节点时为空列表
            else:
                parents = [p.strip() for p in str(parents_raw).split(',')]  # 移除空格

            # 验证父节点是否存在于网络中
            for p in parents:
                if p not in self.variables:
                    raise ValueError(f"父节点 {p} 不在网络中")

            # 验证父节点关系与网络结构一致
            actual_parents = self.get_parents(target_var)
            if sorted(parents) != sorted(actual_parents):
                raise ValueError(
                    f"CPD父节点与网络结构不符：定义 {parents}，实际 {actual_parents}"
                )

            # 提取父节点取值列（匹配格式：父节点名(parent_序号)，无空格）
            parent_cols = [col for col in df.columns if '(parent_' in col]
            # 验证父节点取值列与parents字段的顺序一致性
            expected_parent_cols = [f"{p}(parent_{i + 1})" for i, p in enumerate(parents)]
            if parent_cols != expected_parent_cols:
                raise ValueError(
                    f"父节点列顺序与parents字段不符！\n"
                    f"预期列（按parents顺序）: {expected_parent_cols}\n"
                    f"实际列: {parent_cols}"
                )

            # 生成父节点取值的标准顺序（笛卡尔积）
            parent_value_lists = [self.get_variable_values(p) for p in parents]  # 每个父节点的取值列表
            standard_combinations = list(product(*parent_value_lists))  # 所有组合的标准顺序
            # num_standard_combinations = len(standard_combinations)
            # 解析CSV中的父节点组合
            csv_combinations = []
            for _, row in df.iterrows():
                # 提取每行的父节点取值（按parent_cols顺序）
                combo = tuple(str(row[col]).strip() for col in parent_cols)
                csv_combinations.append(combo)
            # 验证父节点取值的顺序符合标准顺序
            if standard_combinations != csv_combinations:
                raise ValueError(
                    f"父节点取值顺序与标准笛卡尔积的顺序不符！\n"
                    f"标准顺序: {standard_combinations}\n"
                    f"实际顺序: {csv_combinations}"
                )

            # 解析父节点基数（每个父节点的取值数量）
            parent_cardinalities = [len(self.get_variable_values(p)) for p in parents]
            # 目标变量基数
            target_cardinality = len(target_values)
            # 提取概率矩阵并转置（适配pgmpy的TabularCPD格式）
            # 转置后：每行对应目标变量的一个取值，每列对应一组父节点取值组合
            prob_matrix = df[prob_cols].values.T

            # 创建TabularCPD对象
            state_names = {target_var: target_values}
            for parent in parents:
                state_names[parent] = self.get_variable_values(parent)
            cpd = TabularCPD(
                variable=target_var,
                variable_card=target_cardinality,
                values=prob_matrix,
                evidence=parents,
                evidence_card=parent_cardinalities,
                state_names=state_names
            )

            self.model.add_cpds(cpd)
            print(f"成功为变量 {target_var} 添加CPD（文件：{csv_path}）")

        except FileNotFoundError:
            raise FileNotFoundError(f"CPD文件 {csv_path} 不存在")
        except pd.errors.ParserError:
            raise ValueError(f"文件 {csv_path} 格式错误，请检查分号分隔是否正确")
        except IndexError:
            raise ValueError(f"文件 {csv_path} 缺少必要字段（如target_variable、parents等）")
        except Exception as e:
            raise RuntimeError(f"处理CPD时发生错误：{str(e)}")

    @classmethod
    def from_json_file(cls, filename):
        """从JSON文件加载贝叶斯网络"""
        try:
            # 获取JSON文件的绝对路径目录
            file_path = os.path.abspath(filename)
            data_dir = os.path.dirname(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            network = cls()
            network.data_dir = data_dir  # 记录JSON文件所在目录作为数据目录

            # 1. 处理变量组
            variable_groups = data.get('variable_groups', [])
            group_dict = {}  # 组名 -> 变量列表

            for group in variable_groups:
                group_name = group.get('name')
                prefix = group.get('prefix', '')
                suffix = group.get('suffix', '')
                start = group.get('start', 1)
                end = group.get('end', 1)
                values = group.get('values', [])

                if not group_name:
                    raise ValueError("变量组必须包含name字段")

                if not values:
                    raise ValueError(f"变量组 '{group_name}' 必须包含values字段")

                # 生成变量名并添加到网络
                group_vars = []
                for i in range(start, end + 1):
                    if suffix:
                        var_name = f"{prefix}_{i}_{suffix}"
                    else:
                        var_name = f"{prefix}_{i}"

                    network.add_variable(var_name, values)
                    group_vars.append(var_name)

                group_dict[group_name] = group_vars

            # 2. 处理单独的变量
            individual_vars = data.get('individual_variables', {})
            for var_name, values in individual_vars.items():
                if var_name != "_comment":
                    # _comment字段是注释，不是真的变量
                    network.add_variable(var_name, values)

            # 3. 处理边
            edges_data = data.get('edges', {})

            # 3.1 处理单个边
            single_edges = edges_data.get('single_edges', [])
            for edge in single_edges:
                from_name = edge.get('from')
                to_name = edge.get('to')

                if not from_name or not to_name:
                    raise ValueError("边定义必须包含'from'和'to'字段")

                network.add_edge(from_name, to_name)

            # 3.2 处理组边
            group_edges = edges_data.get('group_edges', [])
            for edge in group_edges:
                if 'from_group' in edge and 'to_group' in edge:
                    # 组到组的连接
                    from_group_name = edge['from_group']
                    to_group_name = edge['to_group']
                    connection_type = edge.get('connection_type')

                    if from_group_name not in group_dict:
                        raise ValueError(f"组 '{from_group_name}' 不存在")
                    if to_group_name not in group_dict:
                        raise ValueError(f"组 '{to_group_name}' 不存在")

                    from_vars = group_dict[from_group_name]
                    to_vars = group_dict[to_group_name]

                    if connection_type == 'one_to_one':
                        # 利用zip特性：当两个列表长度不同时，仅匹配到较短列表的长度
                        for from_var, to_var in zip(from_vars, to_vars):
                            network.add_edge(from_var, to_var)
                        # 可选：打印提示信息，说明实际匹配的变量对数
                        matched_count = min(len(from_vars), len(to_vars))
                        if len(from_vars) != len(to_vars):
                            print("一对一连接 - 以变量数少的组为基准，自动忽略长组的多余变量")
                            print(f"'{from_group_name}'与'{to_group_name}'共匹配{matched_count}对变量（忽略长组多余变量）")


                    elif connection_type == 'all_to_all':
                        # 全连接
                        for from_var in from_vars:
                            for to_var in to_vars:
                                network.add_edge(from_var, to_var)

                    else:
                        raise ValueError(f"不支持的连接类型: {connection_type}")

                elif 'from_individual' in edge and 'to_group' in edge:
                    # 单个变量到组的连接
                    from_name = edge['from_individual']
                    to_group_name = edge['to_group']

                    if from_name not in network.variables:
                        raise ValueError(f"变量 '{from_name}' 不存在")

                    if to_group_name not in group_dict:
                        raise ValueError(f"组 '{to_group_name}' 不存在")

                    to_vars = group_dict[to_group_name]
                    for to_var in to_vars:
                        network.add_edge(from_name, to_var)

                elif 'from_group' in edge and 'to_individual' in edge:
                    # 组到单个变量的连接
                    from_group_name = edge['from_group']
                    to_name = edge['to_individual']

                    if from_group_name not in group_dict:
                        raise ValueError(f"组 '{from_group_name}' 不存在")

                    if to_name not in network.variables:
                        raise ValueError(f"变量 '{to_name}' 不存在")

                    from_vars = group_dict[from_group_name]
                    for from_var in from_vars:
                        network.add_edge(from_var, to_name)

                else:
                    raise ValueError("组边定义必须包含有效的from和to组合")

            if network.is_valid_network():
                network.set_model()
            else:
                raise ValueError(f"贝叶斯网络结构数据不符合有向无环图特征")

            return network

        except FileNotFoundError:
            raise FileNotFoundError(f"文件 '{filename}' 不存在")
        except json.JSONDecodeError:
            raise ValueError(f"文件 '{filename}' 格式错误")
        except Exception as e:
            raise RuntimeError(f"加载贝叶斯网络结构数据时发生错误: {str(e)}")


    def print_bn_structure(self):
        try:
            # 获取所有节点（变量）
            all_nodes = self.get_all_variables()
            print(f"共加载 {len(all_nodes)} 个节点，节点关系如下：\n")

            # 遍历每个节点，输出父节点和子节点
            for node in sorted(all_nodes):  # 按节点名排序输出
                parents = self.get_parents(node)
                children = self.get_children(node)

                print(f"节点: {node}")
                print(f"  父节点: {parents if parents else '无'}")
                print(f"  子节点: {children if children else '无'}\n")

        except Exception as e:
            print(f"加载或处理网络时出错: {str(e)}")


def cpd_csv_trans_t1(file_path):
    """
    处理贝叶斯网络CPD文件，将变量名X_i修改为X_i_t1

    Args:
        file_path (str): CPD文件的完整路径

    Returns:
        str: 新生成的文件路径
    """
    # 创建新文件名
    dir_name, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{re.sub(r'X_(\d+)', r'X_\1_t1_de', base_name)}{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # 读取并处理文件内容
    processed_lines = []
    # variable_pattern = re.compile(r'X_?(\d+)')  # 匹配X_i或Xi格式

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过注释行
            if line.strip().startswith('#'):
                processed_lines.append(line)
                continue

            # 修改变量名X_i为X_i_t1
            # 先处理X_i格式
            modified_line = re.sub(r'X_(\d+)', r'X_\1_t1', line)
            # # 再处理Xi格式（如X6, X7）
            # modified_line = re.sub(r'X(\d+)', r'X_\1_t1', modified_line)

            processed_lines.append(modified_line)

    # 保存处理后的内容
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

    return new_file_path
