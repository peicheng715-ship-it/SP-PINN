"""
获取方程 每30h,每断面 的解Q,Z,U,S,A0,S_1   7884  即loss_data需要的真实值
获取S*  7884
获取t_total
计算分组粒径的恢复饱和系数a  7884
计算浑水密度pm,相对密度Delta_p,泥沙密度ps  7884
"""

import numpy as np
import pandas as pd
import data3
import os


def normalize_Q(Q, Q_min, Q_max):
    return (Q - Q_min) / (Q_max - Q_min)

# =================================================================== 获取Q,Z,U,S,A0,S*
# 文件夹路径
folder_path = r'output77/excel'

# 定义存储结果的列表
time_Q_array = []
no_Q_array = []
time_Z_array = []
time_U_array = []
time_S_array = []
time_A0_array = []
time_S_1_array = []
time_D50_array = []

# 按文件名中的数字部分排序
file_names = sorted(
    [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.xlsx')],
    key=lambda x: int(x.split(' ')[-1].split('.')[0])  # 提取数字部分并排序
)
print(file_names)
# 遍历文件夹中的 Excel 文件
for file_name in file_names:
    if file_name.endswith('.xlsx'):  # 确保只处理 Excel 文件
        file_path = os.path.join(folder_path, file_name)

        # 加载 Excel 数据
        df = pd.read_excel(file_path, header=0)

        # 确保列名正确且去除空格
        df.columns = df.columns.map(str).str.strip()

        # 过滤时间为 30 的倍数的行
        filtered_df = df[df["Time(h)"] % 12 == 0]

        # 提取指定列并追加到列表
        time_Q_array.extend(filtered_df[["Time(h)", "QD"]].values.tolist())
        no_Q_array.extend(filtered_df[["QD"]].values.tolist())
        time_Z_array.extend(filtered_df[["Time(h)", "ZW"]].values.tolist())
        time_U_array.extend(filtered_df[["Time(h)", "Uav"]].values.tolist())
        time_S_array.extend(filtered_df[["Time(h)", "S"]].values.tolist())
        time_A0_array.extend(filtered_df[["Time(h)", "dA0/dt(10^-3)"]].values.tolist())
        time_S_1_array.extend(filtered_df[["Time(h)", "S*"]].values.tolist())
        time_D50_array.extend(filtered_df[["Time(h)", "BD50(mm)"]].values.tolist())

# 打印结果的长度作为验证
# print(f"time_S_array length: {len(time_S_array)}")  # 146 * 54 = 7884
# print("流量真实读取：")
# for item in time_A0_array[:300]:
#     print(item)

# 去除第一列时间序列
Q0 = 50
Z0 = 1
U0 = 1
S0 = 1
A0 = 1
a_Q = 1
a_Z = 1
a_U = 1
a_S = 1
a_A0 = 43.2
#======================================================= Q归一化
# no_Q_array = np.array(no_Q_array)
# Q_norm = normalize_Q(no_Q_array,300,1200)
# print("归一化Q：", Q_norm[:200])
#=======================================================
Q_array = [[row[1]] for row in time_Q_array]  # ======================================
Q_array_true = [[row[1]] for row in time_Q_array]   # Q真实值
Q_array = [[round(value[0] / (Q0 * a_Q), 4)] for value in Q_array]   # Q缩放值
Z_array = [[row[1]] for row in time_Z_array]  # ======================================
Z_array_true = [[row[1]] for row in time_Z_array]
Z_array = [[round(value[0] / (Z0 * a_Z), 4)] for value in Z_array]
U_array = [[row[1]] for row in time_U_array]  # ======================================
U_array_true = [[row[1]] for row in time_U_array]
U_array = [[round(value[0] / (U0 * a_U), 4)] for value in U_array]
A0_array = [[row[1]] for row in time_A0_array]  # ====================================
A0_array_true = [[row[1]] for row in time_A0_array]
A0_array = [[round(value[0] * a_A0, 4)] for value in A0_array]
D50_array = [[row[1]] for row in time_D50_array]  # ====================================
D50_array_true = [[row[1]] for row in time_D50_array]

print("Q.len\n", len(Q_array))
# print("Q_array\n", Q_array)
# print("Q_array\n",Q_array)
# min_index = np.argmin(Q_array)
# print("Q_array.min index\n", min_index)
# 如果你还想打印该位置的值
# print("Q_array[min_index]\n", Q_array[min_index])
print("Z.len\n", len(Z_array))
print("U.len\n", len(U_array))
print("A0.len\n", len(A0_array))
print("D50.len\n", len(D50_array))
# print("A0_array:\n" ,A0_array )
# ============================================================================================== 计算分组粒径的恢复饱和系数ak

# 获取各所有的时间步
t_total = [[row[0]] for row in time_S_array]  # ==================
t_true = np.array(t_total)
# print("时间序列\n" , t_true)
T0 = 1
t_total = [[round(value[0] / T0, 4)] for value in t_total]
# print("时间序列\n",t_total)
print("t_total.len:\n", len(t_total))  # 7884
# 去除第一列时间序列
S_array = [[row[1]] for row in time_S_array]  # ====================
# print("含沙量S\n", S_array)
S_1_array = [[row[1]] for row in time_S_1_array]  # ====================
# [[0.78], [0.6], [0.43], [0.4], [0.4], [0.39], [0.36], [0.33]...]

# 转换为np数组
S = np.array(S_array)  # 从第一个断面至最后一个断面的所有时间步数据 含沙量   ===============
S_1 = np.array(S_1_array)  # 挟沙力                                   ===============
delta_S = np.array(data3.last_10S_columns[:19818])  # 从第一个断面至最后一个断面的所有时间步数据 悬沙级配
delta_S1 = np.array(data3.last_10S_1_columns[:19818])  # ===============
# list1_count = len(S_array)
# list2_count = len(delta_S)
# print("对应")
print("S.len:\n", len(S)) # 7884
print("delta_S.len:\n", len(delta_S)) # 7884


# 求分组含沙量和分组挟沙力
S_k = S * delta_S  # =================
S1_k = S_1 * delta_S1  # =================
# print("S为：")
# print(S[:30,:])
# print("delta_S为：")
# print(delta_S[:30,:])
# print("乘积：")
# print(S_k[:30,:])


w_k = data3.last_10w_columns[:19818]
beta = 0.001
# 将列表转换为 NumPy 数组以便矢量化计算
w_k = np.array(w_k)
S_k = np.array(S_k)
S1_k = np.array(S1_k)
print("S1_k.len:\n", len(S1_k))

# # 创建一个掩码，标记 omega_k 中的零值位置
# zero_mask = w_k == 0
#
# # 计算 alpha_k
# alpha_k = np.where(
#     S_k >= S1_k,
#     beta / np.power(w_k, 0.3),  # S_k >= S_{*k}
#     beta / np.power(w_k, 0.7)  # S_k < S_{*k}
# )
#
# # 对 w_k 为零的位置，将 alpha_k 对应的值设置为 0
# alpha_k[zero_mask] = 0  # ======================分组恢复饱和系数
# a = np.array(alpha_k)
# a[:, -5:] = [0.01, 0.02, 0.05, 0.09, 0.15]

# 打印结果
# print("alpha_k:")
# print(len(alpha_k))
# print(alpha_k[0:3,:])
# print("w_k:")
# print(w_k[0:3,:])
# print("s_k和s*_k:")
# print(S_k[0:3,:])
# print(S1_k[0:3,:])
d8 = [0.008, 0.016, 0.031, 0.062, 0.125, 0.250, 0.500, 1.000]
# 将数组转换成 numpy 数组
d8_array = np.array(d8)

# 重复数组以形成 19818 行
d8_array = np.tile(d8_array, (19818, 1))

# 查看矩阵的形状，确认为 (19818, 10)
print(d8_array.shape)
# print("d10_array:", d8_array)

# =======================================================计算浑水密度pm,相对密度Delta_p,泥沙密度ps
Delta_p = 1650
ps = 2650
pm_list = 1000 + (Delta_p / ps) * S
pm = np.array(pm_list)
# print(pm)
print(len(pm))  # 7884
# [[1000.48566038]
#  [1000.37358491]
#  [1000.26773585]
#  ...
#  [1001.94886792]
#  [1001.81811321]
#  [1001.95509434]]
# 7884
# =========================================================== 计算水力坡度Jf
Q = np.array(Q_array)
print("Q.shape\n", Q.shape)
A = data3.A
print("A.shape\n", A.shape)
H = data3.H
print("H.shape\n", H.shape)
n = data3.n
print("n.shape\n", n.shape)
Jf = (Q/A)**2 * n**2 / (H)**(4/3)
# print("Jf:\n", Jf)
print("Jf.shape", Jf.shape)