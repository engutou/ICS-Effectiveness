import os
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

from build_model import MyBayesianNetwork
from build_model import cpd_csv_trans_t1

from effectiveness_calculator import EffectivenessCalculator
from effectiveness_calculator import calculate_statistics
from effectiveness_calculator import plot_efficiency_distribution

from merge_cpds import merge_cpds

from special_bar import plot_dict_bars

def exp1_change_strategy(network: MyBayesianNetwork,
                         calculator: EffectivenessCalculator,
                         observed_states, adfix=False):
    # 定义内部画图函数，参数为result_list
    def _plot_strategy_comparison(data_list, flag, fontsize=20):
        # 全局设置字体大小（根据需求调整数值）
        # plt.rcParams.update({
        #     'font.size': fontsize,  # 全局默认字体大小
        #     'axes.titlesize': fontsize,  # 坐标轴标题大小
        #     'axes.labelsize': fontsize,  # 坐标轴标签大小
        #     'xtick.labelsize': fontsize,  # x轴刻度文字大小
        #     'ytick.labelsize': fontsize,  # y轴刻度文字大小
        #     'legend.fontsize': fontsize,  # 图例文字大小
        #     'figure.titlesize': fontsize  # 总标题大小（若使用suptitle）
        # })
        plt.rcParams["font.size"] = fontsize
        # 解决中文显示问题
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 提取数据
        strategies = [item[0] for item in data_list]  # 策略名称
        means = [item[1] for item in data_list]  # 效能差期望
        entropies = [item[2] for item in data_list]  # 信息熵

        # 创建画布和主轴
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # 绘制效能差期望（柱状图，左轴）
        color = '#4A90E2'  # 稳重蓝（论文常用）
        ax1.set_xlabel('Defence Strategy')
        ax1.set_ylabel('Mean')
        bars = ax1.bar(strategies, means, color=color, alpha=0.8, edgecolor='white', linewidth=1.0, label='Mean')
        ax1.tick_params(axis='y')
        # ax1.tick_params(axis='x', rotation=45)  # 旋转x轴标签，避免重叠
        ax1.tick_params(axis='x')

        # 在柱子上方添加数值标签（精确到小数点后4位）
        for bar, m in zip(bars, means):
            # 获取柱子的位置和高度
            x = bar.get_x() + bar.get_width() / 2  # 柱子中心x坐标
            y = bar.get_height()  # 柱子高度（即mean值）
            # 添加文本标签
            ax1.text(
                x, y,
                f"{m:.4f}",  # 格式化显示4位小数
                ha='center',  # 水平居中
                va='bottom',  # 垂直对齐方式（底部对齐柱子顶部）
                #fontsize=fontsize  # 标签字体大小
            )

        # 创建副轴（共享x轴）
        ax2 = ax1.twinx()
        color = '#E67E22'    # 橙色（对比强）
        marker_color = '#D35400'  # 深橙色点标记
        ax2.set_ylabel('Entropy (bit)')
        ax2.plot(strategies, entropies, color=color, linewidth=1.2, linestyle='--', marker='o', markersize=2, label='Entropy')
        ax2.tick_params(axis='y')

        # 添加标题和图例
        # plt.title('$E(\DeltaF)$ and $H(\DeltaF)$')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # 调整布局并保存
        plt.tight_layout()
        # 确保保存目录存在
        os.makedirs('eleSys', exist_ok=True)
        plt.savefig(f'eleSys/en-strategy_comparison-{flag}.png', dpi=600, bbox_inches='tight')
        print(f"策略对比图已保存至 eleSys/en-strategy_comparison-{flag}.png")
        plt.close()  # 关闭图表释放资源

    eff_additive_13 ={}
    eff_additive_25 = {}
    for ob_flag, states in observed_states.items():
        ob_flag_latex = f'$Z^{{{ob_flag[-1]}}}$'
        evidence_list_A_unknown = []
        evidence_list_A_0 = []
        evidence_list_A_1 = []

        # 无任何防御、不知道有没有攻击
        defend_strategy = {f'D_{i}': '0' for i in range(1, 6)}  # 一共5个防御策略
        evidence = states | defend_strategy
        evidence_list_A_unknown.append(('$D^{0}$', evidence))

        # 无任何防御、无任何攻击
        attack_strategy = {f'A_{i}': '0' for i in range(1, 6)}  # 一共5个攻击策略
        evidence = states | defend_strategy | attack_strategy
        evidence_list_A_0.append(('$D^{0}$', evidence))

        # 无任何防御、有5种攻击
        attack_strategy = {f'A_{i}': '1' for i in range(1, 6)}  # 一共5个攻击策略
        evidence = states | defend_strategy | attack_strategy
        evidence_list_A_1.append(('$D^{0}$', evidence))

        # 只有一种防御策略
        for i in range(1, 6):
            # 不知道有没有攻击
            defend_strategy = {f'D_{j}': '0' for j in range(1, 6)}  # 一共5个防御策略
            defend_strategy[f'D_{i}'] = '1'
            evidence = states | defend_strategy
            evidence_list_A_unknown.append((f'$D^{{{i}}}$', evidence))

            # # 知道没有对应的攻击
            # attack_strategy = {f'A_{i}': '0'}
            # evidence = states | defend_strategy | attack_strategy
            # evidence_list_A_0.append((f'D_{i}=1,A_{i}=0', evidence))
            # 知道没有任何攻击
            attack_strategy = {f'A_{i}': '0' for i in range(1, 6)}  # 一共5个攻击策略
            evidence = states | defend_strategy | attack_strategy
            evidence_list_A_0.append((f'$D^{{{i}}}$', evidence))

            # 知道有对应的攻击
            # attack_strategy = {f'A_{i}': '1'}
            # evidence = states | defend_strategy | attack_strategy
            # evidence_list_A_1.append((f'D_{i}=1,A_{i}=1', evidence))
            # 知道有5种攻击
            attack_strategy = {f'A_{i}': '1' for i in range(1, 6)}  # 一共5个攻击策略
            evidence = states | defend_strategy | attack_strategy
            evidence_list_A_1.append((f'$D^{{{i}}}$', evidence))

        # 增加两个防御策略组合（D1+D3——代表“边界防护+核心加固”, D2+D5——代表“检测+加密防护”）
        # 1+3
        defend_strategy = {f'D_{i}': '0' for i in range(1, 6)}  # 一共5个防御策略
        defend_strategy['D_1'] = '1'
        defend_strategy['D_3'] = '1'
        evidence = states | defend_strategy
        evidence_list_A_unknown.append(('$D^{1,3}$', evidence))

        eff_additive_13[ob_flag_latex] = [['$D^{1,3}$', 1],
                                          ['$D^{{1}}$', 1],
                                          ['$D^{{3}}$', 1]]

        # 知道没有对应的攻击
        # attack_strategy = {'A_1': '0', 'A_3': '0'}
        # evidence = states | defend_strategy | attack_strategy
        # evidence_list_A_0.append((f'D_1_3=1,A_1_3=0', evidence))
        # 知道没有任何攻击
        attack_strategy = {f'A_{i}': '0' for i in range(1, 6)}  # 一共5个攻击策略
        evidence = states | defend_strategy | attack_strategy
        evidence_list_A_0.append(('$D^{1,3}$', evidence))

        # # 知道有对应的攻击
        # attack_strategy = {'A_1': '1', 'A_3': '1'}
        # evidence = states | defend_strategy | attack_strategy
        # evidence_list_A_1.append((f'D_1_3=1,A_1_3=1', evidence))
        # 知道有5种攻击
        attack_strategy = {f'A_{i}': '1' for i in range(1, 6)}  # 一共5个攻击策略
        evidence = states | defend_strategy | attack_strategy
        evidence_list_A_1.append(('$D^{1,3}$', evidence))

        # 2+5
        defend_strategy = {f'D_{i}': '0' for i in range(1, 6)}  # 一共5个防御策略
        defend_strategy['D_2'] = '1'
        defend_strategy['D_5'] = '1'
        evidence = states | defend_strategy
        evidence_list_A_unknown.append(('$D^{2,5}$', evidence))
        eff_additive_25[ob_flag_latex] = [['$D^{2,5}$', 1],
                                          ['$D^{{2}}$', 1],
                                          ['$D^{{5}}$', 1]]

        # attack_strategy = {'A_2': '0', 'A_5': '0'}
        # evidence = states | defend_strategy | attack_strategy
        # evidence_list_A_0.append((f'D_2_5=1,A_2_5=0', evidence))
        # 知道没有任何攻击
        attack_strategy = {f'A_{i}': '0' for i in range(1, 6)}  # 一共5个攻击策略
        evidence = states | defend_strategy | attack_strategy
        evidence_list_A_0.append(('$D^{2,5}$', evidence))

        # attack_strategy = {'A_2': '1', 'A_5': '1'}
        # evidence = states | defend_strategy | attack_strategy
        # evidence_list_A_1.append((f'D_2_5=1,A_2_5=1', evidence))
        # 知道有5种攻击
        attack_strategy = {f'A_{i}': '1' for i in range(1, 6)}  # 一共5个攻击策略
        evidence = states | defend_strategy | attack_strategy
        evidence_list_A_1.append(('$D^{2,5}$', evidence))

        for a_flag, evd_list in zip(['A0', 'A1', 'Ax'], [evidence_list_A_0, evidence_list_A_1, evidence_list_A_unknown]):
            result_list = []
            e0 = 0
            for evidence in evd_list:
                print("作为条件出现的变量")
                print(evidence)
                # 推断后验分布
                # 创建精确推理引擎（变量消除法）
                infer = VariableElimination(network.model)
                pgm_infer_result = infer.query(
                    variables=calculator.var_order,
                    evidence=evidence[1],
                    show_progress=False
                )
                print("推理完成，获得后验联合分布")

                # print(pgm_infer_result.values[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                # print(pgm_infer_result.values[0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0])
                # print(pgm_infer_result.values[0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0])
                # print(pgm_infer_result.values[0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0])
                # print(pgm_infer_result.values[0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0])

                eff_dist = calculator.compute_efficiency_distribution_from_pgm(pgm_infer_result)
                print("效能差的分布计算完成")

                # 计算统计量
                mean, entropy = calculate_statistics(eff_dist)
                print(f"效能均值: {mean:.4f}")
                print(f"信息熵: {entropy:.4f} bit")
                result_list.append((evidence[0], mean, entropy))

                if evidence[0] == '$D^{0}$':
                    e0 = mean
                elif evidence[0] == '$D^{1,3}$':
                    eff_additive_13[ob_flag_latex][0][1] = mean - e0
                elif evidence[0] == '$D^{1}$':
                    eff_additive_13[ob_flag_latex][1][1] = mean - e0
                elif evidence[0] == '$D^{3}$':
                    eff_additive_13[ob_flag_latex][2][1] = mean - e0
                elif evidence[0] == '$D^{2,5}$':
                    eff_additive_25[ob_flag_latex][0][1] = mean - e0
                elif evidence[0] == '$D^{2}$':
                    eff_additive_25[ob_flag_latex][1][1] = mean - e0
                elif evidence[0] == '$D^{5}$':
                    eff_additive_25[ob_flag_latex][2][1] = mean - e0

                # # 绘制效能分布图表
                # plt = plot_efficiency_distribution(eff_dist)
                # plt.savefig(f'eleSys/eff_dist-{ob_flag}-{evidence[0]}-adx.png', dpi=300, bbox_inches='tight')
                # # plt.show()
                # plt.close()

            # 调用内部函数生成对比图
            flag = ob_flag + '-' + a_flag
            if adfix:
                flag += '-adx'
            _plot_strategy_comparison(result_list, flag)
            # todo: delete continue
            # continue

    file_path = 'eleSys/eff_additive-D13.png'
    plot_dict_bars(
        eff_additive_13,
        file_path,
        title=None,
        xlabel="Observed State",
        ylabel="Gain of Effectiveness",
        width=0.25  # 调整柱子宽度使标签更清晰
    )

    file_path = 'eleSys/eff_additive-D25.png'
    plot_dict_bars(
        eff_additive_25,
        file_path,
        title=None,
        xlabel="Observed State",
        ylabel="Gain of Effectiveness",
        width=0.25  # 调整柱子宽度使标签更清晰
    )

def set_bn(adfix=False):
    # 1、从JSON文件加载网络
    filename = "eleSys/eleSys.json"
    network = MyBayesianNetwork.from_json_file(filename)
    # network.print_bn_structure()
    # note:每调用一次generate_all_cpd_csv就会重新生成所有cpd
    # network.generate_all_cpd_csv()
    print("成功从JSON文件加载贝叶斯网络")

    # 2、为贝叶斯网络的所有节点加载cpd
    cpds_to_merge = []
    for var_name in network.get_all_variables():
        # X(t+1)的CPD需要采用有界加法模型进行融合
        if var_name in [f'X_{i}_t1' for i in range(2, 8)]:
            Xt_name = var_name.rstrip('_t1')
            cpd_filename_xt = f"{Xt_name}_cpd.csv"
            cpd_path_xt = os.path.join(network.data_dir, cpd_filename_xt)
            # 处理贝叶斯网络CPD文件，将变量名X_i修改为X_i_t1
            cpd_path_xt1_de = cpd_csv_trans_t1(cpd_path_xt)
            # 只取文件名即可
            data_dir, cpd_path_xt1_de = os.path.split(cpd_path_xt1_de)
            assert data_dir == network.data_dir
            if adfix and var_name not in ('X_6_t1', 'X_7_t1'):  # adx不影响X6、X7
                cpd_path_xt1_ad = cpd_path_xt1_de.replace('_de', '_adx')
            else:
                cpd_path_xt1_ad = cpd_path_xt1_de.replace('_de', '_ad')


            for cpd_file_xt1 in (cpd_path_xt1_de, cpd_path_xt1_ad):
                cpd = network.read_part_cpd_from_csv(cpd_file_xt1)
                cpds_to_merge.append(cpd)
        else:
            csv_filename = f"{var_name}_cpd.csv"
            csv_path = os.path.join(network.data_dir, csv_filename)
            network.add_cpd_from_csv(csv_path)
    cpds_merged = merge_cpds(cpds_to_merge)
    network.model.add_cpds(*cpds_merged)

    if network.model.check_model():
        pass
        # print("成功创建完整的贝叶斯模型，可以开始推理了")
    return network


def set_eff_calculator():
    # 1、初始化效能差计算器
    print("计算效能差的分布")
    num_nodes = 7  # 目标网络的节点数
    network_state_names = [f'X_{i}' for i in range(1, num_nodes + 1)]
    network_state_names_t1 = [f'X_{i}_t1' for i in range(1, num_nodes + 1)]
    target_variables = network_state_names + network_state_names_t1
    # 设置节点的权重
    node_weight_list = [0.2, 0.12, 0.3, 0.16, 0.12, 0.05, 0.05]
    var_weights = {network_state_names[i]: node_weight_list[i] for i in range(0, num_nodes)}
    var_weights_t1 = {network_state_names_t1[i]: node_weight_list[i] for i in range(0, num_nodes)}
    var_weights = var_weights | var_weights_t1
    # 设置每种状态对应的效能
    # note: 必须使用 状态索引 -> 效能
    state2eff = {var: {0: 1.0, 1: 0.8, 2: 0.3, 3: 0.0} for var in target_variables}

    calculator = EffectivenessCalculator(
        var_order=target_variables,
        node_weights=var_weights,
        state_effectiveness=state2eff,
        cache_file_path='delta_e.pkl')
    return calculator


def main():
    """主函数"""
    # ad_fix表示五组条件概率P(X_i (t+1)|X_i (t),A_i,D_i)（i=1,2,…,5）被设为相同的值
    ad_fix = False
    network = set_bn(ad_fix)
    calculator = set_eff_calculator()

    num_nodes = 7
    Z0 = {f'Z_{i}': '0' for i in range(1, num_nodes + 1)}
    observed_states = {'Z=0': Z0,
                       'Z_1=1': Z0 | {'Z_1': '1'},
                       'Z_3=2': Z0 | {'Z_3': '2'},
                       #'Z_13': Z0 | {'Z_1': '1', 'Z_3': '2'}
                      }
    exp1_change_strategy(network, calculator, observed_states, ad_fix)


if __name__ == "__main__":
    import warnings

    # 过滤字体警告
    warnings.filterwarnings(
        "ignore",
        message=".*findfont.*",  # 仅过滤包含关键字findfont的警告
        category=UserWarning
    )

    main()