"""
获取边界BC输入的所有参数
获取观测输入obs的所有参数
组合（x,t,所有参数）以求得lb和ub

"""
import numpy as np

import data3
import data4
import data5
# ===========================================================================BC入口输入参数
x_BC_B = data3.B[:367]
x_BC_A = data3.A[:367]
x_BC_Jf = data4.Jf[:367]
x_BC_hc = data3.hc[:367]
x_BC_w = data3.w[:367]
x_BC_a = data3.a[:367]
x_BC_S_1 = data4.S_1[:367]
x_BC_pm = data4.pm[:367]
x_BC_delta_S = data4.delta_S[:367]
x_BC_delta_S1 = data4.delta_S1[:367]
x_BC_Ax = data5.A_x[:367]

# print("x_Z_BC_Jf:",x_BC_Jf)
print("Shape of x_Z_BC_Jf:", x_BC_Jf.shape)

# =========================================================================BC出口输入参数

x_Z_BC_B = data3.B[::367]
x_Z_BC_A = data3.A[::367]
x_Z_BC_Jf = data4.Jf[::367]
x_Z_BC_hc = data3.hc[::367]
x_Z_BC_w = data3.w[::367]
x_Z_BC_a = data3.a[::367]
x_Z_BC_S_1 = data4.S_1[::367]
x_Z_BC_pm = data4.pm[::367]
x_Z_BC_delta_S = data4.delta_S[::367]
x_Z_BC_delta_S1 = data4.delta_S1[::367]
x_Z_BC_Ax = data5.A_x[::367]

# print("x_Z_BC_a:", x_Z_BC_a)
print("Shape of x_Z_BC_a:", x_Z_BC_a.shape)

# =============================================================================obs观测输入参数A, B, S_1, delta_S1, w, a, Jf, hc, delta_S, pm, Ax
# 原始数据列表
data_A = data3.A_list
data_B = data3.B_list
data_hc = data3.hc_list
data_Jf = data4.Jf.tolist()
data_delta_S1 = data3.delta_S1_list
data_delta_S = data3.delta_S_list
data_w = data3.w_list
data_S1 = data4.time_S_1_array


# 指定地理位置的索引
indices = [10, 19, 37]  # 地理位置索引
num_timesteps = 367  # 每个位置有146个时刻

# 初始化结果列表
x_A_obs = []
x_B_obs = []
x_hc_obs = []
x_Jf_obs = []
x_delta_S1_obs = []
x_delta_S_obs = []
x_w_obs = []
x_S1_obs = []


# 提取每个索引的时刻面积
for idx in indices:
    start = idx * num_timesteps  # 当前地理位置的起始索引
    end = start + num_timesteps  # 当前地理位置的结束索引
    x_A_obs.extend(data_A[start:end])  # 添加当前地理位置的所有数据
    x_B_obs.extend(data_B[start:end])
    x_hc_obs.extend(data_hc[start:end])
    x_Jf_obs.extend(data_Jf[start:end])
    x_delta_S1_obs.extend(data_delta_S1[start:end])
    x_delta_S_obs.extend(data_delta_S[start:end])
    x_w_obs.extend(data_w[start:end])
    x_S1_obs.extend(data_S1[start:end])


# 打印结果
# print("提取的时刻面积列表：", x_A_obs)
print("A长度：\n", len(x_A_obs))
# print("提取的时刻河宽列表：", x_B_obs)
# print("B长度：\n", len(x_B_obs))
# print("提取的时刻hc列表：", x_hc_obs)
# print("hc长度：\n", len(x_hc_obs))
# print("提取的时刻Jf列表：", x_Jf_obs)
print("Jf长度：\n", len(x_Jf_obs))
# print("提取的时刻挟沙力级配列表：", x_delta_S1_obs)
# print("delta_S1长度：\n", len(x_delta_S1_obs))
# print("提取的时刻悬沙级配列表：", x_delta_S_obs)
# print("delta_S长度：\n", len(x_delta_S_obs))
# print("提取的时刻w列表：", x_w_obs)
print("w长度：\n", len(x_w_obs))
# print("提取的时刻S1列表：", x_S1_obs)
# print("S1长度：\n", len(x_S1_obs))


# 假设你的数据是 NumPy 数组
x_a_obs = data3.a
x_Ax_obs = data_Ax = data5.A_x
x_pm_obs = data4.pm
# 每个地理位置有 146 个时刻
num_timesteps = 367

# 指定地理位置索引
indices = [10, 19, 37]  # 地理位置索引

# 提取指定位置的所有时刻数据
x_a_obs = np.vstack([x_a_obs[idx * num_timesteps:(idx + 1) * num_timesteps] for idx in indices])  # 10
x_Ax_obs = np.vstack([x_Ax_obs[idx * num_timesteps:(idx + 1) * num_timesteps] for idx in indices])
x_pm_obs = np.vstack([x_pm_obs[idx * num_timesteps:(idx + 1) * num_timesteps] for idx in indices]) # 1
x_Ax_obs = x_Ax_obs.flatten().reshape(-1, 1)  # 1

# 打印结果
# print("提取的时刻数组：")
# print(x_a_obs)

# print("提取的时刻Ax列表：", x_Ax_obs)
print("Ax长度：", len(x_Ax_obs))
# print("提取的时刻pm列表：", x_pm_obs)
# print("pm长度：", len(x_pm_obs))
x_A_obs = [row[1:2] for row in x_A_obs]
x_B_obs = [row[1:2] for row in x_B_obs]
x_hc_obs = [row[1:2] for row in x_hc_obs]
x_w_obs = [row[1:] for row in x_w_obs]
# x_Jf_obs = [row[1:2] for row in x_Jf_obs]
x_delta_S1_obs = [row[1:] for row in x_delta_S1_obs]
x_delta_S_obs = [row[1:] for row in x_delta_S_obs]
x_S1_obs = [row[1:2] for row in x_S1_obs]



x_A_obs = np.array(x_A_obs)
x_B_obs = np.array(x_B_obs)
x_hc_obs = np.array(x_hc_obs)
x_w_obs = np.array(x_w_obs)
x_Jf_obs = np.array(x_Jf_obs)
x_delta_S1_obs = np.array(x_delta_S1_obs)
x_delta_S_obs = np.array(x_delta_S_obs)
x_S1_obs = np.array(x_S1_obs)


# print("x_Jf_obs:", x_Jf_obs[:400])
print("Shape of A:", x_A_obs.shape)
print("Shape of B:", x_B_obs.shape)
print("Shape of S_1:", x_S1_obs.shape)
print("Shape of delta_S1:", x_delta_S1_obs.shape)
print("Shape of w:", x_w_obs.shape)
print("Shape of a:", x_a_obs.shape)
print("Shape of Jf:", x_Jf_obs.shape)
print("Shape of hc:", x_hc_obs.shape)
print("Shape of delta_S:", x_delta_S_obs.shape)
print("Shape of pm:", x_pm_obs.shape)
print("Shape of Ax:", x_Ax_obs.shape)




