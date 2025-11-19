import numpy as np
import pandas as pd
import random
import warnings
import os
'''
生成扰动数据
'''

def perturb_cpd_by_peak(p0: np.ndarray, peak_change_bound: float) -> np.ndarray:
    """
    按峰值占比波动调整分布

    Args:
        p0: 原始概率分布，二维数组
        peak_change_bound: 调峰边界值，可选2.5%、5%、10%、15%

    Returns:
        调整后的概率分布
    """
    # 确保输入是二维numpy数组
    p0 = np.array(p0)

    # 处理每一行
    result = np.zeros_like(p0)
    for i in range(p0.shape[0]):
        row = p0[i]

        # 检查是否为[1.0, 0, 0, 0]这种某个概率值为1的情况
        if any(prob == 1.0 for prob in row):
            result[i] = row
            continue

        # 为每一行独立生成调峰比例
        if peak_change_bound == 0.025:
            # 对于2.5%，在[-0.025, 0.025]范围内随机，不包括0
            change_ratio = random.uniform(-0.025, 0.025)
            while change_ratio == 0:  # 确保不为0
                change_ratio = random.uniform(-0.025, 0.025)
        else:
            # 对于5%、10%、15%，在[-边界值, -边界值+0.025]和[边界值-0.025, 边界值]范围内随机
            lower_bound = peak_change_bound - 0.025
            if random.random() < 0.5:
                # 正方向调整
                change_ratio = random.uniform(lower_bound, peak_change_bound)
            else:
                # 负方向调整
                change_ratio = random.uniform(-peak_change_bound, -lower_bound)

        peak_idx = np.argmax(row)  # 峰值索引
        original_peak = row[peak_idx]
        other_indices = [j for j in range(len(row)) if j != peak_idx]
        other_sum = row[other_indices].sum()  # 非峰值总和

        # 检查调整是否会超出0~1范围
        new_peak = original_peak * (1 + change_ratio)
        if new_peak < 0 or new_peak > 1:
            result[i] = row
            continue

        # 计算需调整的总量
        delta = new_peak - original_peak  # 正值为增加，需从其他值扣除；负值为减少，需分配给其他值

        # 按比例调整非峰值（保持相对比例）
        other_ratios = row[other_indices] / other_sum if other_sum != 0 else np.ones(len(other_indices)) / len(
            other_indices)
        new_others = row[other_indices] - delta * other_ratios  # delta为正则扣除，为负则增加

        # 确保非峰值不为负（最终约束）
        new_others = np.maximum(new_others, 1e-10)

        # 重组分布并归一化（处理浮点误差）
        p_perturbed = np.zeros_like(row)
        p_perturbed[peak_idx] = new_peak
        p_perturbed[other_indices] = new_others
        p_perturbed = p_perturbed / p_perturbed.sum()

        result[i] = p_perturbed

    return result


def process_cpd_file(input_file, output_file, peak_change_bound):
    """
    处理CPD文件，进行峰值调整并生成新文件

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        peak_change_bound: 调峰边界值
    """
    # 读取CSV文件，忽略注释行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if not line.strip().startswith('#') and line.strip()]

    # 解析表头
    header = lines[0].split(';')

    # 解析数据行
    data_rows = []
    for line in lines[1:]:
        parts = line.split(';')
        data_rows.append(parts)

    # 更准确地识别概率列 - 查找包含等号的列
    prob_columns = []
    prob_indices = []
    for i, col in enumerate(header):
        if '=' in col:
            prob_columns.append(col)
            prob_indices.append(i)

    # 如果没有找到等号列，尝试使用最后4列作为概率列
    if not prob_columns:
        prob_columns = header[-4:]
        prob_indices = list(range(len(header) - 4, len(header)))
        print(f"警告：未找到概率列，使用最后4列: {prob_columns}")

    # 提取概率值部分
    prob_data = []
    for row in data_rows:
        try:
            prob_row = [float(row[i]) for i in prob_indices]
            prob_data.append(prob_row)
        except ValueError as e:
            print(f"错误处理行: {row}")
            print(f"错误详情: {e}")
            # 如果转换失败，尝试使用最后4个元素
            try:
                prob_row = [float(val) for val in row[-4:]]
                prob_data.append(prob_row)
                print(f"使用最后4个元素作为概率值: {prob_row}")
            except ValueError:
                print(f"无法处理此行，跳过: {row}")
                continue

    # 转换为numpy数组
    prob_array = np.array(prob_data)

    # 检查每行概率和是否为1
    for i, row in enumerate(prob_array):
        row_sum = np.sum(row)
        if abs(row_sum - 1.0) > 1e-6:  # 允许小的浮点误差
            warnings.warn(f"第{i + 2}行概率和不为1: {row_sum:.6f}，将进行归一化")
            prob_array[i] = row / row_sum

    # 进行峰值调整
    perturbed_array = perturb_cpd_by_peak(prob_array, peak_change_bound)

    # 创建输出数据
    output_lines = []
    output_lines.append(';'.join(header))

    for i, row in enumerate(data_rows):
        if i >= len(perturbed_array):  # 确保不越界
            break

        # 替换概率值部分
        new_probs = [f"{val:.6f}" for val in perturbed_array[i]]

        # 构建新行 - 保留非概率列，替换概率列
        new_row = []
        prob_idx = 0
        for j, val in enumerate(row):
            if j in prob_indices:
                new_row.append(new_probs[prob_idx])
                prob_idx += 1
            else:
                new_row.append(val)

        output_lines.append(';'.join(new_row))

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"处理完成！调整后的CPD已保存到: {output_file}")
    print(f"处理了 {len(perturbed_array)} 行数据")


# 使用示例
if __name__ == "__main__":
    # input_dir = "eleSys/cpd_origin/"
    # output_dir = "eleSys/cpd_perturb/1/150/"
    # if not os.path.exists(output_dir): os.mkdir(output_dir)
    # peak_change_bound = 0.15  # 2.5%调峰边界值
    #
    # # 列出文件夹下所有文件和子文件夹的名称
    # all_items = os.listdir(input_dir)
    #
    # # 筛选出文件（排除子文件夹）
    # for item in all_items:
    #     if 'de' in item: continue
    #     item_path = os.path.join(input_dir, item)  # 拼接完整路径
    #     if os.path.isfile(item_path):  # 判断是否为文件
    #         process_cpd_file(item_path, output_dir+item.split('.csv')[0]+'_perturb_150.csv', peak_change_bound)
    '''
    再生成9个扰动数据
    '''
    for i in range(2, 11):
        for ratio in ['25', '50', '100', '150']:
            input_dir = "eleSys/cpd_origin/"
            output_dir = f"eleSys/cpd_perturb/{i}/{ratio}/"
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            peak_change_bound = int(ratio)/1000  # 2.5%调峰边界值

            # 列出文件夹下所有文件和子文件夹的名称
            all_items = os.listdir(input_dir)

            # 筛选出文件（排除子文件夹）
            for item in all_items:
                if 'de' in item: continue
                item_path = os.path.join(input_dir, item)  # 拼接完整路径
                if os.path.isfile(item_path):  # 判断是否为文件
                    process_cpd_file(item_path, output_dir + item.split('.csv')[0] + f'_perturb_{ratio}.csv', peak_change_bound)
