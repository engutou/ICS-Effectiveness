from pgmpy.factors.discrete import TabularCPD
import re

def read_cpds_from_file_XYZ(filename, pattern):
    cpds = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割每个表格
    tables = re.findall(pattern, content, re.DOTALL)
    
    for table in tables:
        lines = [line.strip() for line in table.split('\n') if line.strip() and not line.startswith('#')]
        # 解析父节点和子节点名称（例如 X1 和 Y1）
        header = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        parent = header[0]  # 如 X1
        child = header[1]   # 如 Y1
        
        # 收集所有条件概率
        data = []
        for line in lines[2:]:  # 跳过表头和分隔行
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(cells) < 3:
                continue
            x_val = int(cells[0])
            y_val = int(cells[1])
            prob = float(cells[2].split('#')[0].strip())  # 去除注释
            data.append((x_val, y_val, prob))
        
        # 提取所有可能的父节点和子节点取值
        x_values = sorted({x for x, _, _ in data})
        y_values = sorted({y for _, y, _ in data})
        
        # 构建条件概率表
        cpd_values = []
        for x in x_values:
            row = [0.0] * len(y_values)
            for y in y_values:
                # 查找匹配的概率，若不存在则默认为0.0
                prob = next((p for xx, yy, p in data if xx == x and yy == y), 0.0)
                row[y_values.index(y)] = prob
            cpd_values.append(row)
        
        # 创建 TabularCPD 对象
        cpd = TabularCPD(
            variable=child,
            variable_card=len(y_values),
            evidence=[parent],
            evidence_card=[len(x_values)],
            values=[[p for p in row] for row in zip(*cpd_values)],
            state_names={parent: [str(item) for item in x_values],
                         child: [str(item) for item in y_values]}
        )
        cpds.append(cpd)
    
    return cpds


from pgmpy.factors.discrete import TabularCPD
import re


def read_cpds_from_file_X(filename, is_t1=False):
    cpds = []
    with open(filename, 'r') as f:
        content = f.read()

    # 分割所有表格（以##开头的注释行作为分隔）
    tables = re.findall(r'##.*?(?=##|\Z)', content, re.DOTALL)

    for table in tables:
        # 提取表格内容
        lines = [line.strip() for line in table.split('\n') if line.strip() and not line.startswith('#')]
        if not lines:
            continue

        # 解析表头
        header = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        prob_col = next((col for col in header if col.startswith('P(')), None)

        # 提取变量关系
        if prob_col:
            # 解析P(Child|Parents)结构
            match = re.match(r'P\(([A-Za-z]\d+)[∣|](.*)\)', prob_col)
            if match:
                child = match.group(1)
                parents = [p.strip() for p in match.group(2).split(',')] if match.group(2) else []
            else:
                # 先验概率表
                if is_t1:  # t+1时刻的状态变量中，不需要先验概率
                    continue
                child = header[0]
                parents = []

        if is_t1:
            # t+1时刻的状态变量，变量名需要调整
            child = child + '_t1'
            parents = [parent + '_t1' for parent in parents]

        # 收集所有变量取值和概率
        data = {}
        for line in lines[2:]:  # 跳过表头和分隔线
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if not cells:
                continue

            # 提取值和概率（处理注释）
            *var_values, prob = cells
            prob = float(prob.split('#')[0].strip())

            # 构建键：父节点值的元组（如果有）
            key = tuple(int(v) for v in var_values[:-1]) if parents else ()
            data[key] = data.get(key, {})
            data[key][int(var_values[-1])] = prob

        # 构建CPD参数
        variable = child
        variable_card = len(data[()]) if not parents else len(next(iter(data.values())))

        # 提取父节点信息
        evidence = parents
        evidence_card = []
        state_names = {variable: sorted(data[()].keys())} if not parents else {}

        # 处理有父节点的情况
        if parents:
            # 获取所有父节点取值组合
            parent_values = set(key for key in data.keys())
            sorted_parent_combinations = sorted(parent_values)

            # 构建values矩阵
            values = []
            child_values = set()  # 顺便记录child节点的状态名称
            for parent_comb in sorted_parent_combinations:
                sorted_child_values = sorted(data[parent_comb].keys())
                values.append([data[parent_comb][v] for v in sorted_child_values])

                child_values = child_values.union(sorted_child_values)
                state_names[child] = sorted(child_values)

            # 转置为pgmpy需要的格式
            values = [[p for p in row] for row in zip(*values)]

            # 获取父节点基数
            evidence_card = [len(set(pv[i] for pv in parent_values)) for i in range(len(parents))]

            # 为父节点构建状态名称
            for i, parent in enumerate(parents):
                state_names[parent] = sorted(set(pv[i] for pv in parent_values))

            # # 为子节点构建状态名称
            # child_values = set()
            # for value_dict in data.values():
            #     child_values = child_values.union(set(value_dict.keys()))
            # state_names[child] = sorted(child_values)

        else:
            # 先验概率直接作为values
            values = [[v[1]] for v in sorted(data[()].items())]


        # state_names设置为str
        for v, n in state_names.items():
            state_names[v] = [str(item) for item in n]

        # 创建CPD对象
        cpd = TabularCPD(
            variable=variable,
            variable_card=variable_card,
            evidence=evidence,
            evidence_card=evidence_card,
            values=values,
            state_names=state_names
        )
        cpds.append(cpd)

    return cpds


# 示例用法
if __name__ == "__main__":
    pattern_y = r'\|\s*X\d\s*\|\s*Y\d\s*\|\s*P\(Y\d∣X\d\)\s*\|.*?(?=\n\n|\Z)'
    cpds_y = read_cpds_from_file_XYZ(r"C:\Users\zhiyo\Desktop\PHD效能研究\pY_given_X.txt", pattern_y)
    for cpd in cpds_y:
        print(cpd)
        print("\n" + "-"*50 + "\n")

    pattern_z = r'\|\s*X\d\s*\|\s*Z\d\s*\|\s*P\(Z\d∣X\d\)\s*\|.*?(?=\n\n|\Z)'
    cpds_z = read_cpds_from_file_XYZ(r"C:\Users\zhiyo\Desktop\PHD效能研究\pZ_given_X.txt", pattern_z)
    for cpd in cpds_z:
        print(cpd)
        print("\n" + "-" * 50 + "\n")

    cpds = read_cpds_from_file_X(r"C:\Users\zhiyo\Desktop\PHD效能研究\pX.txt")
    for cpd in cpds:
        print(cpd)
        print("\n" + "-" * 50 + "\n")
