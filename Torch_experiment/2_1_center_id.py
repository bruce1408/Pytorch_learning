import numpy as np

def mean_col_by_row_attribute(seg):
    # 确保输入的 seg 是一个二维数组
    assert (len(seg.shape) == 2)

    # 找到 seg 中所有大于 0 的唯一值（中心ID）
    center_ids = np.unique(seg[seg > 0])
    lines = []
    ids = []
    
    # 遍历每个中心ID
    for idx, cid in enumerate(center_ids):
        cols, rows = [], []
        
        # 遍历每一行
        for y_op in range(seg.shape[0]):
            # 找到当前行中等于当前中心ID的所有列索引
            x_op = np.where(seg[y_op, :] == cid)[0]
            
            # 如果找到的列索引不为空
            if x_op.size > 0:
                # 计算这些列索引的平均值
                x_op = np.mean(x_op)
                # 将平均值添加到 cols 中，对应的行索引 y_op 添加到 rows 中
                cols.append(x_op)
                rows.append(y_op)
        
        # 将当前中心ID对应的（列平均值列表，行索引列表）添加到 lines 中
        lines.append((cols, rows))
        # 将当前中心ID添加到 ids 中
        ids.append(cid)
    
    return lines, ids


if __name__ == "__main__":

    seg = np.array([
        [0, 1, 1, 0],
        [0, 2, 0, 0],
        [2, 2, 0, 0],
        [0, 0, 0, 0]
    ])

    res = mean_col_by_row_attribute(seg)
    print(res[0], res[1])