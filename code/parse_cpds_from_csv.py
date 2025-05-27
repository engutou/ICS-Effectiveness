from pgmpy.factors.discrete import TabularCPD


def parse_cpd_from_csv(file_path):
    """
    解析包含条件概率表的文件，生成 pgmpy 的 CPD 对象列表
    note: 当前版本需确保输入数据已按笛卡尔积顺序排列

    Args:
        file_path (str): 数据文件路径

    Returns:
        list: 包含 TabularCPD 对象的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]  # 读取并过滤空行

    cpd_sections = []
    current_section = []
    for line in lines:
        if line.startswith('=='):  # 条件概率表标题行（分隔符）
            if current_section:
                cpd_sections.append(current_section)
                current_section = []
        else:
            current_section.append(line)
    if current_section:  # 处理最后一个表
        cpd_sections.append(current_section)

    cpds = []
    for section in cpd_sections:
        # 解析表头行（以 # 开头的行）
        header_line = [line for line in section if line.startswith('#')][0]
        header = header_line[1:].strip().split(',')  # 去除 # 并按逗号分割
        if any(not h.strip() for h in header):
            raise ValueError(f"表头包含空字段: {header}")

        # 解析数据行
        data_lines = [line.split(',') for line in section if not line.startswith('#')]
        data = []
        for dl in data_lines:
            stripped = [d.strip() for d in dl]
            if any(not s for s in stripped):
                raise ValueError(f"数据行包含空字段: {dl}")
            data.append(stripped)

        # 提取变量名和证据变量
        variable = header[-2]  # 倒数第二列为目标变量（如，X1_t1）
        evidence_vars = header[:-2]  # 前 n-2 列为条件变量（如 X1, A1, D1）

        # 提取变量取值状态（去重排序）
        def get_states(column_data):
            return sorted(list(set(column_data)))

        variable_states = get_states([row[-2] for row in data])
        evidence_states = [get_states([row[i] for row in data]) for i in range(len(evidence_vars))]

        # 构建状态到索引的映射
        # var_state_map = {state: idx for idx, state in enumerate(variable_states)}
        # evidence_state_maps = [{state: idx for idx, state in enumerate(states)}
        #                        for states in evidence_states]

        # 生成概率矩阵（按证据变量顺序排列）
        # num_evidence = len(evidence_vars)
        evidence_card = [len(states) for states in evidence_states]
        variable_card = len(variable_states)

        # 按证据变量组合顺序排列数据（需确保输入数据已按笛卡尔积顺序排列）
        probs = []
        for row in data:
            prob = float(row[-1])
            probs.append(prob)

        # 调整概率数组形状以匹配 pgmpy 要求（证据变量优先排列）
        values = [probs[i * variable_card: (i + 1) * variable_card]
                  for i in range(len(probs) // variable_card)]
        values = [list(row) for row in zip(*values)]

        # 创建 CPD 对象
        cpd = TabularCPD(
            variable=variable,
            variable_card=variable_card,
            evidence=evidence_vars,
            evidence_card=evidence_card,
            values=values,
            state_names={
                variable: variable_states,
                **{ev: evidence_states[i] for i, ev in enumerate(evidence_vars)}
            }
        )
        cpds.append(cpd)

    return cpds


# 使用示例
if __name__ == "__main__":
    file_path = r"C:\Users\zhiyo\Desktop\PHD效能研究\pXt1_given_Xt_At_Dt.csv"  # 替换为实际文件路径
    try:
        cpds = parse_cpd_from_csv(file_path)
        for idx, cpd in enumerate(cpds, 1):
            print(f"条件概率表 {idx}:")
            print(cpd)
            print("\n" + "-" * 50)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")