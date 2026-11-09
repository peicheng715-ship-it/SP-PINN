"""
整理完整输入(x,t) 输出结果 按每个断面的所有时间排列
整理BC输入 输出
计算A_x
"""
import numpy as np
import pandas as pd
import data3
import data4


# =======================================================================整理完整输入(x,t) 输出结果
# 完整输入(x,t)============
x_total = data3.x_total
t_total = data4.t_total
if len(x_total) != len(t_total):
    raise ValueError("x 和 t 的长度不一致，无法一一对应组合。")
# 组合二维列表
X_f = [[x_total[i][0], t_total[i][0]] for i in range(len(x_total))]
X_f = np.array(X_f)  # 也就是X_start
# 输出结果
print("入口X：\n", X_f)
print("入口X_f.len:\n", len(X_f))
lb = X_f.min(0)
ub = X_f.max(0)
# print("lb=", lb)
# print("ub=", ub)
# ====================================================================组合所有输入、参数  获取最小最大值lb、ub   后续进行归一化
# 完整输出==========
Q_star = data4.Q_array
# Q_star = data4.Q_norm
Q_star_true = data4.Q_array_true
Z_star = data4.Z_array
Z_star_true = data4.Z_array_true
U_star = data4.U_array
U_star_true = data4.U_array_true
S_star = data4.S_array
A0_star = data4.A0_array
A0_star_true = data4.A0_array_true
D50_star = data4.D50_array

Q_star = np.array(Q_star)
Q_star_true = np.array(Q_star_true)
Z_star = np.array(Z_star)
Z_star_true = np.array(Z_star_true)
H_star = data3.H
U_star = np.array(U_star)
U_star_true = np.array(U_star_true)
S_star = np.array(S_star)
A0_star = np.array(A0_star)
A0_star_true = np.array(A0_star_true)
D50_star = np.array(D50_star)

# print("缩放Q：\n", Q_star)
# print("缩放A0：\n", Q_star)
# print("真实Q：\n", Q_star_true)
average_value = np.mean(Z_star_true)
print("Qmin：\n",min(Q_star))
print("Qmax：\n",max(Q_star))
# print("Z_star_true 的平均值是:", average_value)
# ==========================================================================整理BC输入 输出 367
# 获取入口断面BC输入
X_u_BC = X_f[:367]
X_h_BC = X_f[:367]

# 获取出口断面BC输入
X_Z_BC = X_f[::367]

# 输出结果
# print("入口BC输入：\n", X_u_BC)
# print("出口BC输入：\n", X_Z_BC)

# 获取入口断面BC边界Q,S
Q_BC = Q_star[:367]
S_BC = S_star[:367]

# 获取出口断面BC边界Z
Z_BC = Z_star[::367]
H_BC = H_star[::367]


Q_BC = np.array(Q_BC)
S_BC = np.array(S_BC)
Z_BC = np.array(Z_BC)
H_BC = np.array(H_BC)
print("Q_BC：\n",max(Q_BC))
print("入口Q：\n", Q_BC)
# print("入口S：\n", S_BC)
# print("入口Z：\n", Z_BC)
# print("入口X_BC：\n", X_u_BC)
# print("出口X_Z_BC：\n", X_Z_BC)
print("出口H_BC：\n", H_BC)
# print("H.shape", H_star.shape)
# ==========================================================================计算A_t,A_x
import numpy as np

# 读取Excel文件
file_path = r'output77/每12h参数新.xlsx'  # 替换为你的文件路径
sheet_name = 0  # 如果有多个sheet，可替换为对应sheet名称或索引

# 读取数据
df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)


# 筛选出有效数据行
# 定义有效数据条件：非空值且包含指定列名称
def is_valid_row(row):
    try:
        # 将所有值转换为浮点数，如果失败则认为是无效行
        [float(val) for val in row if not pd.isna(val)]
        return True
    except ValueError:
        return False


# 应用过滤规则
valid_rows = df.apply(is_valid_row, axis=1)
data = df[valid_rows]

# 添加列名，替换为你需要的列名
data.columns = [
    "Ncs",
    "DistLG(km)",
    "Hav(m)",
    "Uav(m/s)",
    "Area(m2)",
    "BW(m)",
    "QD(m3/s)",
    "ZW(m)",
    "ZBav(m)",
    "Froude",
    "RN",
    "J(%%)",
    "Qmc%",
    "挟沙力级配1",
    "挟沙力级配2",
    "挟沙力级配3",
    "挟沙力级配4",
    "挟沙力级配5",
    "挟沙力级配6",
    "挟沙力级配7",
    "挟沙力级配8",
    "挟沙力级配9",
    "挟沙力级配10",
    "悬沙级配1",
    "悬沙级配2",
    "悬沙级配3",
    "悬沙级配4",
    "悬沙级配5",
    "悬沙级配6",
    "悬沙级配7",
    "悬沙级配8",
    "悬沙级配9",
    "悬沙级配10",
    "床沙级配1",
    "床沙级配2",
    "床沙级配3",
    "床沙级配4",
    "床沙级配5",
    "床沙级配6",
    "床沙级配7",
    "床沙级配8",
    "床沙级配9",
    "床沙级配10",
    "浑水沉速1",
    "浑水沉速2",
    "浑水沉速3",
    "浑水沉速4",
    "浑水沉速5",
    "浑水沉速6",
    "浑水沉速7",
    "浑水沉速8",
    "浑水沉速9",
    "浑水沉速10",
    "恢复饱和系数1",
    "恢复饱和系数2",
    "恢复饱和系数3",
    "恢复饱和系数4",
    "恢复饱和系数5",
    "恢复饱和系数6",
    "恢复饱和系数7",
    "恢复饱和系数8",
    "恢复饱和系数9",
    "恢复饱和系数10",
    "形心水深hc"
]

# 选择需要的列
filtered_data_A = data[["DistLG(km)", "Area(m2)"]]
filtered_data_B = data[["DistLG(km)", "BW(m)"]]
filtered_data_hc = data[["DistLG(km)", "形心水深hc"]]
# filtered_data_Jf = data[["DistLG(km)", "水力坡度Jf"]]
# 协沙力级配
filtered_data_delta_S1 = data[
    ["DistLG(km)", "挟沙力级配1", "挟沙力级配2", "挟沙力级配3", "挟沙力级配4", "挟沙力级配5", "挟沙力级配6",
     "挟沙力级配7", "挟沙力级配8", "挟沙力级配9", "挟沙力级配10"]]  #
# 悬沙级配
filtered_data_delta_S = data[
    ["DistLG(km)", "悬沙级配1", "悬沙级配2", "悬沙级配3", "悬沙级配4", "悬沙级配5", "悬沙级配6",
     "悬沙级配7", "悬沙级配8", "悬沙级配9", "悬沙级配10"]]  #
filtered_data_w = data[
    ["DistLG(km)", "浑水沉速1", "浑水沉速2", "浑水沉速3", "浑水沉速4", "浑水沉速5", "浑水沉速6",
     "浑水沉速7", "浑水沉速8", "浑水沉速9", "浑水沉速10"]]  #
# 查看结果
# print(filtered_data_A.head(200).to_string())
# print("tets:",filtered_data_A[:10])

df = pd.DataFrame(filtered_data_A)

# 定义行数间隔
chunk_size = 93 # 93个断面

# 分割 DataFrame 每 54 行形成二维列表  有146个
result = [df.iloc[i:i + chunk_size].values.tolist() for i in range(0, len(df), chunk_size)]

# 查看结果
# for idx, chunk in enumerate(result):
#     print(f"二维列表 {idx + 1}:")
#     print(chunk)
# 二维列表 1: [[129.67, 1525.7], [134.37, 974.2], [135.82, 1764.7], [136.47, 1166.3], [137.87, 1076.3], [138.32, 1374.5]...]


result = np.array(result)  # 确保转换完成
print(result.shape)  # 检查形状是否符合预期

# 存储导数结果
derivatives = []

# 遍历每个二维数组
for batch in result:
    X = batch[:, 0]  # 提取 X 列
    A = batch[:, 1]  # 提取 A 列

    # 计算数值导数 (dA/dX) 使用 np.diff
    dA_dX = np.diff(A) / np.diff(X)  # 差分计算导数

    # 处理边界的导数
    # 使用前两个点的差分计算第一个点的导数
    dA_dX = np.insert(dA_dX, 0, (A[1] - A[0]) / (X[1] - X[0]))

    # 使用最后两个点的差分计算最后一个点的导数
    # dA_dX = np.append(dA_dX, (A[-1] - A[-2]) / (X[-1] - X[-2]))

    # 存储结果
    derivatives.append(dA_dX)

# 转换为 numpy 数组
derivatives = np.array(derivatives)

# 查看结果
# for i, dA_dX in enumerate(derivatives):
#     print(f"批次 {i + 1} 的导数：")
#     print(dA_dX)
#
# print(len(derivatives))

# 提取每列数据
A_x_54 = [derivatives[:, i].tolist() for i in range(derivatives.shape[1])]

# 输出结果
# for idx, lst in enumerate(A_x_54):
#     print(f"列表 {idx + 1}：{lst}")

# print(len(A_x_54))  # 54

A_x = np.array(A_x_54)
print("A_x",A_x)
print("A_x.shape", A_x.shape)
A_x = A_x.flatten()
A_x = A_x[:, np.newaxis]
A_x = A_x[:19818]
print(len(A_x))
# print(A_x[0:300])   # 7884
