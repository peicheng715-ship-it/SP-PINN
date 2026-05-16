"""
读取A,B,y,
"""
import numpy as np
import pandas as pd

# ================================================================================================= 坐标（x）
# 读取 Excel 文件（将 'your_file.xlsx' 替换成你的实际文件名）
df = pd.read_excel(r'data/table.xlsx', usecols=[0], skiprows=2)

# 删除空行（即两列中至少有一个为空的行）
df_cleaned = df.dropna()

# 将 x 和 t 列转换成嵌套列表形式 [[x1], [x2], ...]
x_list = df_cleaned.iloc[:, 0].astype(float).map(lambda v: [v]).tolist()

x_total = np.array(x_list)
# print("x[102]", x_list[:102])

x_arr = np.array(x_total)
print("x.shape", x_arr.shape)

# ====================================================================================================== 时间t
df = pd.read_excel(r'data/sediment.xlsx')

# 提取第一列
col = df.iloc[:, 0]
col_repeated = np.tile(col, 21)
t_arr = col_repeated.reshape(-1, 1)
print("t.shape", t_arr.shape)
# print("t[102]", t_arr[:102])

# =======================================================================================================拼接（x，t）
xt_combined = np.hstack((x_arr, t_arr))
# # 使用 np.hstack 横向拼接两个数组
xt_list = xt_combined.tolist()
# print("xt_combined[102]", xt_combined[:102])
print("xt_combined.shape", xt_combined.shape)

# # ================================================================================================== 流量Q
df = pd.read_excel(r'data/table.xlsx', usecols=[2], skiprows=2)

# 删除空行（即两列中至少有一个为空的行）
df_cleaned = df.dropna()

# 将 x 和 t 列转换成嵌套列表形式 [[x1], [x2], ...]
Q_list = df_cleaned.iloc[:, 0].astype(float).map(lambda v: [v]).tolist()

Q = np.array(Q_list)
Q = Q / 500
Q_list = Q.tolist()
print("Q.shape", Q.shape)
print("Q[102]:", Q[0:102])
# # ================================================================================================= 悬沙浓度S
df = pd.read_excel(r'data/sediment.xlsx')
# 提取断面位置（第一行的列名，转换为 float）
x_positions = [float(col) for col in df.columns]
x_positions = x_positions[1:]
data = df.values

# 1. 构建 [[x, y], ...] 结构，按断面顺序展开
xS_data = []
S_data_list = []
for j, x in enumerate(x_positions):
    for i in range(data.shape[0]):
        y = data[i, j + 1]
        xS_data.append([x, y])
        S_data_list.append([y])

S_data = np.array(S_data_list)
S_data = S_data / 100
S_data_list = S_data.tolist()
# print("S[:102]:", S_data[:102])
print("S.shape", S_data.shape)
# # ================================================================================================= 水深H
df = pd.read_excel(r'data/depth.xlsx')
# 提取断面位置（第一行的列名，转换为 float）
x_positions = [float(col) for col in df.columns]
x_positions = x_positions[1:]
data = df.values

# 1. 构建 [[x, y], ...] 结构，按断面顺序展开
xH_data = []
H_data_list = []
for j, x in enumerate(x_positions):
    for i in range(data.shape[0]):
        y = data[i, j + 1]
        xH_data.append([x, y])
        H_data_list.append([y])

H = np.array(H_data_list)
H_list = H.tolist()
H = np.array(H_data_list)
# print("S[:102]:", S_data[:102])
print("H.shape", H.shape)
# # =============================================================================================== 浑水沉速W [后续 *分组级配]
df = pd.read_excel(r'data/fall velocity.xlsx')
data = df.values

# 1. 构建 [[x, y], ...] 结构，按断面顺序展开
xW_data = []
W_data_list = []
for j, x in enumerate(x_positions):
    for i in range(data.shape[0]):
        y = data[i, j + 1]
        xW_data.append([x, y])
        W_data_list.append([y])

W = np.array(W_data_list)

# print("W_data[:102]:", W[:102])
print("W.shape", W.shape)
# ================================================================================================= 过水面积 A
df = pd.read_excel(r'data/table.xlsx', usecols=[5], skiprows=2)

# 删除空行（即两列中至少有一个为空的行）
df_cleaned = df.dropna()

# 将 x 和 t 列转换成嵌套列表形式 [[x1], [x2], ...]
A_list = df_cleaned.iloc[:, 0].astype(float).map(lambda v: [v]).tolist()

A = np.array(A_list)
print("A.shape", A.shape)
# print("A[102]:", A[0:102])
# # ================================================================================================ 水面宽度B
df = pd.read_excel(r'data/table.xlsx', usecols=[6], skiprows=2)

# 删除空行（即两列中至少有一个为空的行）
df_cleaned = df.dropna()

# 将 x 和 t 列转换成嵌套列表形式 [[x1], [x2], ...]
B_list = df_cleaned.iloc[:, 0].astype(float).map(lambda v: [v]).tolist()

B = np.array(B_list)
print("B.shape", B.shape)
# print("B[102]:", B[0:102])
# =================================================================================================== 曼宁系数 n
n = 0.035
# ================================================================================================ 恢复饱和系数[按粒径] a 7组
a = [0.01, 0.01, 0.007, 0.01, 0.01, 0.01, 0.011]
repeated_coeffs = [a[:] for _ in range(2100)]  # 使用列表推导式复制
a_list = repeated_coeffs
a = np.tile(a, (2100, 1))  # 形状 (2100, 7)
# print("恢复饱和系数:", a[:20])
print("a.shape", a.shape)
# =================================================================================================== 水流挟沙力 S*
S_1 = 0.3

# # =================================================================================================== 悬沙级配 delta_S按流量分级
delta_S_100 = [0.4, 0.3, 0.2, 0.08, 0.01, 0.01, 0.01]
delta_S_1000 = [0.5, 0.2, 0.2, 0.08, 0.01, 0.01, 0.01]
delta_S_5000 = [0.55, 0.18, 0.15, 0.08, 0.02, 0.01, 0.01]

# 构造对应的 delta_S 列表
delta_S_all_list = []
Q0 = 500

for q in Q_list:
    val = q[0]  # 提取单个值
    if 0 <= val < (100/Q0):
        delta_S_all_list.append(delta_S_100[:])
    elif (100/Q0) <= val < (1000/Q0):
        delta_S_all_list.append(delta_S_1000[:])
    elif (1000/Q0) <= val <= (8000/Q0):
        delta_S_all_list.append(delta_S_5000[:])
    else:
        raise ValueError(f"Q值 {val} 超出预期范围 [0, 5000]")

delta_S_all = np.array(delta_S_all_list)
print("delta_S_all.shape", delta_S_all.shape)
# print(delta_S_all[:20])  # 看看前3个结果
# print(f"总共生成了 {len(delta_S_all)} 条 delta_S")

# ================================================================================================ 重力加速度 g
g = 9.81

# ================================================================================================ 床沙干密度 P
P = 1300

# # ================================================================================================ 冲淤面积 A0
df = pd.read_excel(r'data/A0.xlsx')
data = df.values

xA0_data = []
A0_data_list = []
for j, x in enumerate(x_positions):
    for i in range(data.shape[0]):
        y = data[i, j + 1]
        xA0_data.append([x, y])
        A0_data_list.append([y])

A0_data = np.array(A0_data_list)
A0_data_list = [[x[0] / 500] for x in A0_data_list]
A0_data = A0_data / 500
print("A0.shape", A0_data.shape)
# print("y_data[:102]:", A0_data[:102])
# # ================================================================================================== A_x
# 每100个元素分割成一个子列表
split_lists = [A_list[i:i+100] for i in range(0, len(A_list), 100)]

# 打印分割后的结果
# for idx, sublist in enumerate(split_lists):
#     print(f"列表 {idx + 1}: {sublist}")

split_lists_clean = [[item[0] for item in sublist] for sublist in split_lists]

# 转置：从 (21, 100) 变成 (100, 21)
new_lists = [list(x) for x in zip(*split_lists_clean)]

# 打印看看
# for idx, lst in enumerate(new_lists):
#     print(f"第{idx+1}个新列表：{lst}")

x = np.arange(0, 500 * 21, 500)  # [0, 500, 1000, ..., 10000]

# 存放每个列表求导后的结果
all_derivatives = []

for lst in new_lists:
    # 先去除括号，比如如果lst是[[90.11], [129.68], ...]的话
    if isinstance(lst[0], list):
        lst = [item[0] for item in lst]

    # 计算偏导，使用 np.gradient
    derivative = np.gradient(lst, x)  # 对x求导
    all_derivatives.append(derivative.tolist())
# 打印
# print(all_derivatives)
transposed = list(map(list, zip(*all_derivatives)))
# print(transposed)

flattened = [item for sublist in transposed for item in sublist]
# 再每个元素套一层[]
A_x = [[x] for x in flattened]
A_x = np.array(A_x)
print("A_x.shape", A_x.shape)  # 应该是 2100
# print(A_x[:200])   # 看前5个