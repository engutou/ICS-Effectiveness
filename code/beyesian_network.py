from nltk import entropy
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from itertools import product
import numpy as np
from typing import Literal
import pickle
import os

from torch.backends.opt_einsum import strategy

from build_cpds_from_file import read_cpds_from_file_XYZ, read_cpds_from_file_X
from merge_cpds import merge_cpds
from parse_cpds_from_csv import parse_cpd_from_csv
from effectiveness_calculator import EffectivenessCalculator, plot_efficiency_distribution
from plot_test import plot_efficiency_metrics

# 定义变量名
var_names_Xt = [f'X{i}' for i in range(1, 5)]  # X(t) = (X_1(t), X_2(t), X_3(t), X_4(t))
var_names_Xt1 = [f'X{i}_t1' for i in range(1, 5)]  # X(t+1) = (X_1(t+1), X_2(t+1), X_3(t+1), X_4(t+1))
var_names_Yt = [f'Y{i}' for i in range(1, 5)]  # Y(t) = (Y_1(t), Y_2(t), Y_3(t), Y_4(t))
var_names_Zt = [f'Z{i}' for i in range(1, 5)]  # Z(t) = (Z_1(t), Z_2(t), Z_3(t), Z_4(t))
var_names_At = [f'A{j}' for j in range(1, 8)]  # A(t) = (A_1(t), ..., A_7(t))
var_names_Dt = [f'D{j}' for j in range(1, 8)]  # D(t) = (D_1(t), ..., D_7(t))

# 定义节点的可能取值
state_names = {}
for var_names in (var_names_Xt, var_names_Xt1, var_names_Yt, var_names_Zt):
    for var in var_names:
        # 各设备的状态含义
        # 'X1': ['正常', '轻度受损', '严重受损', '瘫痪'],
        # 'X2': ['正常', '轻度堵塞', '严重堵塞', '故障停机'],
        # 'X3': ['正常', '数据偏差', '数据篡改', '故障无数据'],
        # 'X4': ['正常', '流量波动', '流量超限', '完全失控']
        state_names[var] = ['0', '1', '2', '3']  # 设备状态取值为：0、1、2、3,统一为str型
for var_names in (var_names_At, var_names_Dt):
    for var in var_names:
        state_names[var] = ['0', '1']  # 攻击或防守行为的取值为：'0'、'1'


def save_to_pickle(data, filename):
    """将数据保存到pickle文件"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已成功保存到 {filename}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


def load_from_pickle(filename):
    """从pickle文件加载数据"""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件 {filename} 不存在")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None


def build_bn():
    # 定义贝叶斯网络结构
    bn_edges = []
    # X(t) --> X(t+1)
    # X(t) --> Y(t)、Z(t)
    assert len(var_names_Xt) == len(var_names_Xt1)
    assert len(var_names_Xt) == len(var_names_Yt)
    assert len(var_names_Xt) == len(var_names_Zt)
    for i, X in enumerate(var_names_Xt):
        bn_edges.append((var_names_Xt[i], var_names_Xt1[i]))
        bn_edges.append((var_names_Xt[i], var_names_Yt[i]))
        bn_edges.append((var_names_Xt[i], var_names_Zt[i]))

    # X(t) --> A(t)、D(t) --> X(t+1)
    bn_edges.extend([('X1', 'A1'), ('A1', 'X1_t1')])  # 中央控制服务器
    bn_edges.extend([('X1', 'D1'), ('D1', 'X1_t1')])
    bn_edges.extend([('X1', 'A2'), ('A2', 'X1_t1')])  # 中央控制服务器
    bn_edges.extend([('X1', 'D2'), ('D2', 'X1_t1')])
    bn_edges.extend([('X2', 'A3'), ('A3', 'X2_t1')])  # 水泵
    bn_edges.extend([('X2', 'D3'), ('D3', 'X2_t1')])
    bn_edges.extend([('X2', 'A4'), ('A4', 'X2_t1')])  # 水泵
    bn_edges.extend([('X2', 'D4'), ('D4', 'X2_t1')])
    bn_edges.extend([('X3', 'A5'), ('A5', 'X3_t1')])  # 水质传感器
    bn_edges.extend([('X3', 'D5'), ('D5', 'X3_t1')])
    bn_edges.extend([('X4', 'A6'), ('A6', 'X4_t1')])  # 流量控制器
    bn_edges.extend([('X4', 'D6'), ('D6', 'X4_t1')])
    bn_edges.extend([('X4', 'A7'), ('A7', 'X4_t1')])  # 流量控制器
    bn_edges.extend([('X4', 'D7'), ('D7', 'X4_t1')])
    # # X(t) --> A(t)、D(t)
    # bn_edges.append(('X1', 'A1'))  # 中央控制服务器
    # bn_edges.append(('X1', 'D1'))
    # bn_edges.append(('X1', 'A2'))  # 中央控制服务器
    # bn_edges.append(('X1', 'D2'))
    # bn_edges.append(('X2', 'A3'))  # 水泵
    # bn_edges.append(('X2', 'D3'))
    # bn_edges.append(('X2', 'A4'))  # 水泵
    # bn_edges.append(('X2', 'D4'))
    # bn_edges.append(('X3', 'A5'))  # 水质传感器
    # bn_edges.append(('X3', 'D5'))
    # bn_edges.append(('X4', 'A6'))  # 流量控制器
    # bn_edges.append(('X4', 'D6'))
    # bn_edges.append(('X4', 'A7'))  # 流量控制器
    # bn_edges.append(('X4', 'D7'))

    # X(t)、X(t+1)内部各分量之间的依赖关系
    for var_group in (var_names_Xt, var_names_Xt1):
    # for var_group in (var_names_Xt,):
        bn_edges.extend([(var_group[2], var_group[0]),  # X3-->X1
                         (var_group[0], var_group[1]),  # X1-->X2
                         (var_group[0], var_group[3]),  # X1-->X4
                         (var_group[1], var_group[3])  # X2-->X4
                         ])
    # 创建贝叶斯网络模型
    model = DiscreteBayesianNetwork(bn_edges)
    return model


def convert_pgmpy_to_joint_dist(pgmpy_result, var_names):
    """
    将pgmpy的联合概率分布结果转换为脚本所需的格式

    参数:
    - pgmpy_result: VariableElimination.infer()返回的结果
    - state_names: 二元组，格式为((当前时刻状态变量名), (下一时刻状态变量名))
                  例如: (('X1','X2','X3','X4'), ('X1_t1','X2_t1','X3_t1','X4_t1'))

    返回:
    - 适配脚本的联合分布字典 {(x_t, x_t1): probability}
    """
    # 校验参数格式
    if not isinstance(var_names, tuple) or len(var_names) != 2:
        raise ValueError("state_names必须是包含两个元素的二元组")

    state_names_t, state_names_t1 = var_names

    # 校验变量名与pgmpy结果是否一致
    result_vars = pgmpy_result.variables
    expected_vars = state_names_t + state_names_t1

    if result_vars != expected_vars:
        if len(set(result_vars)) != len(set(expected_vars)):
            missing = [v for v in expected_vars if v not in result_vars]
            extra = [v for v in result_vars if v not in expected_vars]

            error_msg = "变量不匹配: "
            if missing:
                error_msg += f"缺少变量: {missing}; "
            if extra:
                error_msg += f"多余变量: {extra}"
        else:
            error_msg = "变量顺序不匹配"

        raise ValueError(error_msg)

    joint_dist = {}

    # 获取变量的状态名称映射
    var_state_names = pgmpy_result.state_names

    # 获取各变量的状态空间大小
    var_dims = pgmpy_result.cardinality

    # 生成所有可能的状态组合索引
    indices = np.ndindex(*var_dims)

    # 遍历所有可能的状态组合
    for idx in indices:
        # 构建状态名称元组
        state_tuple = tuple(
            var_state_names[var][state_idx]
            for var, state_idx in zip(result_vars, idx)
        )

        # 分割为当前时刻和下一时刻的状态
        states_t = state_tuple[:len(state_names_t)]
        states_t1 = state_tuple[len(state_names_t):]

        # 获取对应的概率值
        prob = pgmpy_result.values[idx]

        # 添加到联合分布字典
        joint_dist[(states_t, states_t1)] = prob

    return joint_dist


# 执行精确推理计算后验概率P(X(t+1),X(t) | Y(t),A(t))和P(X(t+1),X(t) | Z(t),D(t))
def perform_exact_inference(model, player: Literal['attacker', 'a', 'defender', 'd']):
    if player in ('attacker', 'a'):
        observed_var = 'Y'
        strategy_var = 'A'
    elif player in ('defender', 'd'):
        observed_var = 'Z'
        strategy_var = 'D'
    else:
        raise ValueError(f'player参数取值不合法：{player}')
    # 初始化效能计算实例
    calculator = EffectivenessCalculator()

    # 创建精确推理引擎（变量消除法）
    infer = VariableElimination(model)

    variables = var_names_Xt + var_names_Xt1
    all_net_states = list(product(*[state_names[v] for v in var_names_Xt]))  # 4个设备的所有可能状态
    # all_net_states = [('0', '0', '0', '0')]

    state_to_eff_dist = {}
    for one_net_state in all_net_states:
        strategy_states = ['0'] * len(var_names_At)
        for i in range(len(strategy_states) + 1):
            current_strategy = strategy_states.copy()
            if i > 0:
                current_strategy[i - 1] = '1'  # 攻击/防守状态

            print(f'推断联合后验概率P(X(t+1), X(t) | {observed_var}(t)={one_net_state}, {strategy_var}(t)={player[0]}{i})')

            evidence = {}
            for j in range(1, len(one_net_state) + 1):
                evidence[f'{observed_var}{j}'] = one_net_state[j-1]  # 攻击者观测到的网络状态为Y(t)
            for j in range(1, len(current_strategy) + 1):
                evidence[f'{strategy_var}{j}'] = current_strategy[j-1]  # 攻击者明确知道攻击策略

            # 推断后验概率
            pgm_infer_result = infer.query(
                variables=variables,
                evidence=evidence,
                show_progress=False
            )

            joint_dist = convert_pgmpy_to_joint_dist(pgm_infer_result, (var_names_Xt, var_names_Xt1))

            # 计算效能分布
            eff_dist = calculator.compute_efficiency_distribution(joint_dist)
            mean, entropy = calculator.calculate_statistics(eff_dist)
            state_to_eff_dist[(one_net_state, f'{strategy_var}{i}')] = (eff_dist, mean, entropy)
            print(f'完成推断.....')

    return state_to_eff_dist


def set_cpds_xt1():
    cpds = []
    cpds_intra = read_cpds_from_file_X(r"C:\Users\zhiyo\Desktop\PHD效能研究\pX.txt", is_t1=True)
    # for cpd in cpds_intra:
    #     print(cpd.variable)
    #     print(cpd.get_evidence())
    cpds.extend(cpds_intra)

    cpds_inter = parse_cpd_from_csv(r"C:\Users\zhiyo\Desktop\PHD效能研究\pXt1_given_Xt_At_Dt.csv")
    # for cpd in cpds_inter:
    #     print(cpd.variable)
    #     print(cpd.get_evidence())
    cpds.extend(cpds_inter)

    cpds = merge_cpds(cpds)
    return cpds


# 示例使用
if __name__ == "__main__":
    filename1 = 'attacker_effectiveness.pkl'
    filename2 = 'defender_effectiveness.pkl'
    attacker_eff, defender_eff = {}, {}
    if not os.path.exists(filename1) or not os.path.exists(filename2):
        m = build_bn()
        # # 构建X(t)内部各变量间的条件概率表
        # build_cpds_Xt(m)
        cpds_x = read_cpds_from_file_X(r"C:\Users\zhiyo\Desktop\PHD效能研究\pX.txt")
        m.add_cpds(*cpds_x)

        # P(Y(t)|X(t))
        pattern_y = r'\|\s*X\d\s*\|\s*Y\d\s*\|\s*P\(Y\d∣X\d\)\s*\|.*?(?=\n\n|\Z)'
        cpds_y = read_cpds_from_file_XYZ(r"C:\Users\zhiyo\Desktop\PHD效能研究\pY_given_X.txt", pattern_y)
        m.add_cpds(*cpds_y)

        # P(Z(t)|X(t))
        pattern_z = r'\|\s*X\d\s*\|\s*Z\d\s*\|\s*P\(Z\d∣X\d\)\s*\|.*?(?=\n\n|\Z)'
        cpds_z = read_cpds_from_file_XYZ(r"C:\Users\zhiyo\Desktop\PHD效能研究\pZ_given_X.txt", pattern_z)
        m.add_cpds(*cpds_z)

        # P(A(t)|X(t))
        pattern_a = r'\|\s*X\d\s*\|\s*A\d\s*\|\s*P\(A\d∣X\d\)\s*\|.*?(?=\n\n|\Z)'
        cpds_a = read_cpds_from_file_XYZ(r"C:\Users\zhiyo\Desktop\PHD效能研究\pA_given_X.txt", pattern_a)
        m.add_cpds(*cpds_a)

        # P(D(t)|X(t))
        pattern_d = r'\|\s*X\d\s*\|\s*D\d\s*\|\s*P\(D\d∣X\d\)\s*\|.*?(?=\n\n|\Z)'
        cpds_d = read_cpds_from_file_XYZ(r"C:\Users\zhiyo\Desktop\PHD效能研究\pD_given_X.txt", pattern_d)
        m.add_cpds(*cpds_d)

        cpds_xt1 = set_cpds_xt1()
        m.add_cpds(*cpds_xt1)

        # 验证模型有效性
        if m.check_model():
            print("贝叶斯网络模型有效")
        else:
            print("模型无效，请检查CPD设置")
            for cpd in m.get_cpds():
                print(f"CPD检查: {cpd.variable} - {m.get_cpds(cpd.variable).validate()}")

        # 计算后验概率
        if not os.path.exists(filename1):
            attacker_eff = perform_exact_inference(m, player='a')
            save_to_pickle(attacker_eff, filename1)

        if not os.path.exists(filename2):
            defender_eff = perform_exact_inference(m, player='d')
            save_to_pickle(defender_eff, filename2)
    else:
        attacker_eff = load_from_pickle(filename1)
        defender_eff = load_from_pickle(filename2)


    for k, eff_dist in attacker_eff.items():
        # 绘制效能分布图表
        plt = plot_efficiency_distribution(eff_dist[0])
        one_net_state, strategy_var = k[0], k[1]
        file_name = f'Yt-{'_'.join(one_net_state)}-{strategy_var}_eff_dist.png'
        print(file_name)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    for k, eff_dist in defender_eff.items():
        # 绘制效能分布图表
        plt = plot_efficiency_distribution(eff_dist[0])
        one_net_state, strategy_var = k[0], k[1]
        file_name = f'Zt-{'_'.join(one_net_state)}-{strategy_var}_eff_dist.png'
        print(file_name)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()


    all_net_states = list(product(*[state_names[v] for v in var_names_Xt]))  # 4个设备的所有可能状态
    for one_net_state in all_net_states:
        strategy_states = ['0'] * len(var_names_At)
        mean_list_a, entr_list_a = [], []
        mean_list_d, entr_list_d = [], []
        for i in range(len(strategy_states)+1):
            current_strategy = strategy_states.copy()
            if i > 0:
                current_strategy[i-1] = '1'  # 实施了一种攻击或者防御策略

            _, mean, entr = attacker_eff[(one_net_state, f'A{i}')]
            mean_list_a.append(mean)
            entr_list_a.append(entr)

            _, mean, entr = defender_eff[(one_net_state, f'D{i}')]
            mean_list_d.append(mean)
            entr_list_d.append(entr)
        # 绘制图表
        plt = plot_efficiency_metrics(
            mean_list_a, entr_list_a,
            mean_list_d, entr_list_d,
            strategies=["A0/D0", "A1/D1", "A2/D2", "A3/D3", "A4/D4", "A5/D5", "A6/D6", "A7/D7"],
            title="mean and entropy of effectiveness over different strategies",
            save_path=r'C:\Users\zhiyo\Desktop\PHD效能研究\fig' + f'\\strategy8-{'_'.join(one_net_state)}.png'
        )

        # 显示图表
        # plt.show()
        plt.close()

