def merge_cpds_by_clipped_add(*P_list, m):
    """
    计算多个条件概率表按照有界加法模型合并后的条件概率表 P(X|Y1,Y2,...Yk;Z1,Z2,...,Zl;...)

    参数:
    *P_list: 多个条件概率表，每个表为字典，键为 (a, condition_tuple)，值为概率
    m: 随机变量 X 的最大取值

    返回:
    字典，条件概率表 P(X|Y1,Y2,...Yk;Z1,Z2,...,Zl;...)，键为 (n, condition1_tuple, condition2_tuple, ...)，值为概率
    """
    # 初始化结果概率表
    P = {}

    # 获取所有条件组合
    all_condition_sets = []
    for P_i in P_list:
        condition_set = set(condition for (a, condition) in P_i.keys())
        all_condition_sets.append(condition_set)

    from itertools import product
    # 对每个可能的条件组合进行计算
    for condition_tuple in product(*all_condition_sets):
        # 对每个 n ∈ {0,1,...,m} 计算概率
        for n in range(m + 1):
            total_prob = 0.0

            from itertools import product
            # 遍历所有可能的 a1, a2, ... 组合
            all_possible_values = [range(m + 1) for _ in P_list]
            for values in product(*all_possible_values):
                if sum(values) < n:
                    continue  # 总和小于 n，跳过
                if min(sum(values), m) == n:
                    # 获取 P1(a1|Y), P2(a2|Z), ...
                    prob_product = 1.0
                    for i, (a, P_i, condition) in enumerate(zip(values, P_list, condition_tuple)):
                        p = P_i.get((a, condition), 0.0)
                        prob_product *= p
                    total_prob += prob_product

            # 存储结果
            key = (n,) + condition_tuple
            P[key] = total_prob

    return P


def cpd_to_dict(cpd):
    """
    将 TabularCPD 转换为字典表示

    参数:
    cpd: TabularCPD 对象

    返回:
    dict: 嵌套元组为键的条件概率表，格式为：
          (variable值, (evidence1值, evidence2值, ...)) -> 概率值
    """
    result = {}
    var = cpd.variable
    var_card = cpd.variable_card
    evidence = cpd.get_evidence()
    evidence = evidence[::-1]
    evidence_card = list(cpd.cardinality)
    evidence_card = evidence_card[1:]

    # 获取状态名称映射
    state_names = cpd.state_names

    # 生成所有可能的证据组合
    from itertools import product
    evidence_states = list(product(
        *[range(card) for card in evidence_card]
    ))

    # 遍历每个证据组合
    for ev_idx, ev_state in enumerate(evidence_states):
        # 构建证据值元组（按证据顺序）
        ev_values = tuple(
            state_names[evidence[i]][ev_state[i]]
            for i in range(len(evidence))
        )

        # 获取该证据组合下的条件概率分布
        probs = cpd.get_values()[:, ev_idx]

        # 遍历变量的每个可能值
        for var_val_idx in range(var_card):
            # 获取变量的实际值（通过状态名称映射）
            var_val = state_names[var][var_val_idx]

            # 构建键
            key = (var_val, ev_values)

            # 存储概率值
            result[key] = probs[var_val_idx]

    return result



def test_cpd_to_dict():
    from pgmpy.factors.discrete import TabularCPD
    # 创建示例CPD
    cpd = TabularCPD(
        variable='X', variable_card=2,
        # 正确形状：(2, 12) = (X状态数, Y状态数×Z状态数)
        values=[
            [0.7, 0.6, 0.5, 0.4,  # Y=0, Z取不同值时X=0的概率
             0.3, 0.2, 0.2, 0.1,  # Y=1, Z取不同值时X=0的概率
             0.1, 0.1, 0.05, 0.01],  # Y=2, Z取不同值时X=0的概率

            [0.3, 0.4, 0.5, 0.6,  # Y=0, Z取不同值时X=1的概率
             0.7, 0.8, 0.8, 0.9,  # Y=1, Z取不同值时X=1的概率
             0.9, 0.9, 0.95, 0.99]  # Y=2, Z取不同值时X=1的概率
        ],
        evidence=['Y', 'Z'],  # 证据变量列表
        evidence_card=[3, 4],  # Y有3个状态，Z有4个状态
        state_names={
            'X': ['x0', 'x1'],
            'Y': ['y0', 'y1', 'y2'],
            'Z': ['z0', 'z1', 'z2', 'z3']
        }
    )

    # 转换为字典
    cpd_dict = cpd_to_dict(cpd)
    print(cpd_dict)


# 示例使用
if __name__ == "__main__":
    # k=3, l=2 的示例
    m = 2  # X 的最大取值为 2

    # 条件概率表 P1(X|Y1,Y2,Y3)
    # Y1, Y2, Y3 各有两个状态 {0,1}
    P1 = {
        # Y=(0,0,0)
        (0, (0, 0, 0)): 0.7, (1, (0, 0, 0)): 0.2, (2, (0, 0, 0)): 0.1,
        # Y=(0,0,1)
        (0, (0, 0, 1)): 0.6, (1, (0, 0, 1)): 0.3, (2, (0, 0, 1)): 0.1,
        # Y=(0,1,0)
        (0, (0, 1, 0)): 0.5, (1, (0, 1, 0)): 0.4, (2, (0, 1, 0)): 0.1,
        # Y=(0,1,1)
        (0, (0, 1, 1)): 0.4, (1, (0, 1, 1)): 0.5, (2, (0, 1, 1)): 0.1,
        # Y=(1,0,0)
        (0, (1, 0, 0)): 0.3, (1, (1, 0, 0)): 0.5, (2, (1, 0, 0)): 0.2,
        # Y=(1,0,1)
        (0, (1, 0, 1)): 0.2, (1, (1, 0, 1)): 0.6, (2, (1, 0, 1)): 0.2,
        # Y=(1,1,0)
        (0, (1, 1, 0)): 0.1, (1, (1, 1, 0)): 0.7, (2, (1, 1, 0)): 0.2,
        # Y=(1,1,1)
        (0, (1, 1, 1)): 0.0, (1, (1, 1, 1)): 0.8, (2, (1, 1, 1)): 0.2,
    }

    # 条件概率表 P2(X|Z1,Z2)
    # Z1, Z2 各有两个状态 {0,1}
    P2 = {
        # Z=(0,0)
        (0, (0, 0)): 0.6, (1, (0, 0)): 0.3, (2, (0, 0)): 0.1,
        # Z=(0,1)
        (0, (0, 1)): 0.5, (1, (0, 1)): 0.4, (2, (0, 1)): 0.1,
        # Z=(1,0)
        (0, (1, 0)): 0.4, (1, (1, 0)): 0.5, (2, (1, 0)): 0.1,
        # Z=(1,1)
        (0, (1, 1)): 0.3, (1, (1, 1)): 0.6, (2, (1, 1)): 0.1,
    }

    # 条件概率表 P3(X|W1)
    # W1 有两个状态 {0,1}
    P3 = {
        (0, (0,)): 0.8, (1, (0,)): 0.1, (2, (0,)): 0.1,
        (0, (1,)): 0.2, (1, (1,)): 0.7, (2, (1,)): 0.1,
    }

    # 计算线性叠加模型下的条件概率表
    P = merge_cpds_by_clipped_add(P1, P2, P3, m=m)

    # 打印部分结果示例
    example_conditions = [
        (0, (0, 0, 0), (0, 0), (0,)),
        (0, (1, 1, 0), (1, 1), (1,)),
        (1, (1, 0, 1), (0, 1), (1,)),
        (2, (1, 1, 1), (1, 1), (1,)),
    ]

    for n, *conditions in example_conditions:
        key = (n,) + tuple(conditions)
        print(f"P(X={n}|Y={conditions[0]},Z={conditions[1]},W={conditions[2]}) = {P[key]:.6f}")