from pgmpy.factors.discrete import TabularCPD
from itertools import product
from typing import List  # 兼容3.8及其以下版本

def group_cpds(cpds: List[TabularCPD]):
    """
    将输入列表中所有evidence非空的cpd按照variable分组

    参数:
    cpds: list，包含TabularCPD对象的列表

    返回:
    dict: 键为variable名称，值为对应的cpd列表
    """
    grouped_cpds = {}
    for cpd in cpds:
        # 检查cpd的evidence是否非空
        if cpd.get_evidence():
            var = cpd.variable
            if var not in grouped_cpds:
                grouped_cpds[var] = []
            grouped_cpds[var].append(cpd)

    return grouped_cpds


def inconsistent_values(names, values):
    """
    判断相同变量的取值是否存在不一致

    参数:
    names: List[str]，变量名称列表
    values: List[any]，变量取值列表

    返回:
    bool: 若存在相同变量但取值不同，返回True；否则返回False
    """
    var_dict = {}

    for name, value in zip(names, values):
        if name in var_dict:
            if var_dict[name] != value:
                return True
        else:
            var_dict[name] = value

    return False


def merge_duplicate_variables(names, values):
    """
    合并重复变量并保留首次出现的值

    参数:
    names: List[str]，变量名称列表
    values: List[any]，变量取值列表

    返回:
    tuple: 合并后的变量名称列表和对应值列表

    # 示例测试
    names = ['X', 'Y', 'Z', 'X', 'Y']
    values = [1, 2, 3, 4, 5]
    merged_names, merged_values = merge_duplicate_variables(names, values)

    print("合并后的变量名称:", merged_names)  # 输出: ['X', 'Y', 'Z']
    print("合并后的变量取值:", merged_values)  # 输出: [1, 2, 3]

    """
    merged_names = []
    merged_values = []
    seen = set()

    for name, value in zip(names, values):
        if name not in seen:
            seen.add(name)
            merged_names.append(name)
            merged_values.append(value)

    return merged_names, merged_values


def merge_cpds_by_clipped_add(cpd_list: List[TabularCPD]):
    """
    计算多个 TabularCPD 基于有界加模型融合后的 TabularCPD P(X|Y1,Y2,...Yk;Z1,Z2,...,Zl;...)

    参数:
    cpd_list: List[TabularCPD]，多个 TabularCPD 对象

    返回:
    TabularCPD，合并后的条件概率表
    """
    # 校验输入的所有 cpd 的 variable 名称、variable 的 state_name、variable card 是一致的
    first_cpd = cpd_list[0]
    for cpd in cpd_list[1:]:
        if cpd.variable != first_cpd.variable:
            raise ValueError("所有 CPD 的 variable 名称必须一致。")
        if cpd.state_names[cpd.variable] != first_cpd.state_names[first_cpd.variable]:
            raise ValueError("所有 CPD 的 variable 的 state_name 必须一致。")
        if cpd.cardinality[0] != first_cpd.cardinality[0]:
            raise ValueError("所有 CPD 的 variable card 必须一致。")

    # 允许条件变量有交集
    # 但是条件变量有交集的时候，相同条件变量的取值不能冲突
    # 否则，条件不成立，概率为0
    all_evidence_vars = []
    for cpd in cpd_list:
        all_evidence_vars.extend(cpd.variables[1:])

    # 合并所有变量的state_names
    state_names = {}
    for cpd in cpd_list:
        state_names.update(cpd.state_names)


    # 随机变量 X 的最大取值不通过参数传递，直接从 cpd 中获取
    m = first_cpd.cardinality[0] - 1

    # 获取所有条件变量的名称和状态数
    condition_vars = []
    condition_card = []
    for cpd in cpd_list:
        condition_vars.extend(cpd.variables[1:])
        condition_card.extend(cpd.cardinality[1:])

    # 初始化结果概率表
    result_prob = {}

    # 获取所有条件组合
    all_condition_states = list(product(*[range(card) for card in condition_card]))

    # 对每个可能的条件组合进行计算
    for condition_state in all_condition_states:
        # 校验条件组合是否冲突，如果冲突，直接将概率设置为0
        if inconsistent_values(condition_vars, condition_state):
            # 存储结果
            for n in range(m + 1):
                key = (n,) + tuple(condition_state)
                result_prob[key] = None
            continue

        # 对每个 n ∈ {0,1,...,m} 计算概率
        for n in range(m + 1):
            total_prob = 0.0

            # 遍历所有可能的 a1, a2, ... 组合
            all_possible_values = [range(m + 1) for _ in cpd_list]
            for values in product(*all_possible_values):  # values是在各个条件概率表中variable的取值
                if sum(values) < n:
                    continue  # 总和小于 n，跳过
                if min(sum(values), m) == n:
                    # 获取 P1(a1|Y), P2(a2|Z), ...
                    prob_product = 1.0
                    start_index = 0
                    # for i, (a, cpd) in enumerate(zip(values, cpd_list)):
                    for a, cpd in zip(values, cpd_list):
                        num_conditions = len(cpd.variables) - 1
                        condition = condition_state[start_index:start_index + num_conditions]
                        prob = cpd.values[(a,) + condition]
                        prob_product *= prob
                        start_index += num_conditions
                    total_prob += prob_product

            # 存储结果
            key = (n,) + tuple(condition_state)
            result_prob[key] = total_prob

    # 构建合并后的 TabularCPD
    # result_card = [m + 1] + condition_card
    new_condition_vars, new_condition_card = merge_duplicate_variables(condition_vars, condition_card)
    result_values = []
    for n in range(m + 1):
        n_values = []
        for condition_state in all_condition_states:
            key = (n,) + tuple(condition_state)
            if result_prob[key] is not None:  # todo: 这里的逻辑对吗...
                n_values.append(result_prob[key])
        result_values.append(n_values)
    result_cpd = TabularCPD(variable=first_cpd.variable, variable_card=m + 1,
                            values=result_values, evidence=new_condition_vars,
                            evidence_card=new_condition_card, state_names=state_names)

    return result_cpd


def merge_cpds(cpd_list: List[TabularCPD]):
    return_cpds = []
    grouped = group_cpds(cpd_list)
    for variable, cpd_group in grouped.items():
        print(f'合并{variable}的{len(cpd_group)}个条件概率表')
        merged = merge_cpds_by_clipped_add(cpd_group)
        return_cpds.append(merged)
    return return_cpds
