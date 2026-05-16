"""
BC边界条件
"""
import numpy as np
import pandas as pd

import data1

N0 = 100

# ===================================================== 上游边界输入BC_X  下游边界输入BC_Z_X 上游边界条件BC_Q  下游边界条件BC_Z
BC_X = data1.xt_combined[:N0]

# print("BC_X:", BC_X[:, 0:1])
BC_Z_X = data1.xt_combined[::N0]
# print("BC_Z_X:", BC_Z_X)

BC_Q = data1.Q[:N0]
BC_Z = data1.H[::N0]
# print("BC_Z:", BC_Z)
# ============================================================================== BC入口及出口参数 A, B, w, a, delta_S
x_BC_A = data1.A[0:N0]
x_BC_B = data1.B[0:N0]
x_BC_w = data1.W[0:N0]
x_BC_a = data1.a[0:N0]
x_BC_delta_S = data1.delta_S_all[0:N0]

# print("x_BC_A:", x_BC_A)
# print("x_BC_B:", x_BC_B)
# print("x_BC_w:", x_BC_w)
# print("x_BC_delta_S:", x_BC_delta_S)
# print("x_BC_a:", x_BC_a)

x_BC_Z_A = data1.A[::N0]
x_BC_Z_B = data1.B[::N0]
x_BC_Z_w = data1.W[::N0]
x_BC_Z_a = data1.a[::N0]
x_BC_Z_delta_S = data1.delta_S_all[::N0]

# print("x_BC_Z_A:", x_BC_Z_A)
# print("x_BC_Z_B:", x_BC_Z_B)
# print("x_BC_Z_w:", x_BC_Z_w)
# print("x_BC_delta_Z_S:", x_BC_Z_delta_S)
# print("x_BC_Z_a:", x_BC_Z_a)
# ==============================================================================  观测数据obs
X_obs = []
X_obs_Q = []
X_obs_Z = []
X_obs_S = []
X_obs_A0 = []
X_obs_A = []
X_obs_B = []
X_obs_w = []
X_obs_a = []
X_obs_delta_S = []
obs_list = [7]
for obs in obs_list:
    obs = obs * N0
    X_obs.append(data1.xt_list[obs:obs+N0])
    X_obs_Q.append(data1.Q_list[obs:obs+N0])
    X_obs_Z.append(data1.H_list[obs:obs+N0])
    X_obs_S.append(data1.S_data_list[obs:obs + N0])
    X_obs_A0.append(data1.A0_data_list[obs:obs + N0])
    X_obs_A.append(data1.A_list[obs:obs+N0])
    X_obs_B.append(data1.B_list[obs:obs+N0])
    X_obs_w.append(data1.W_data_list[obs:obs+N0])
    X_obs_a.append(data1.a_list[obs:obs+N0])
    X_obs_delta_S.append(data1.delta_S_all_list[obs:obs+N0])


X_obs = [item for sublist in X_obs for item in sublist]
X_obs_Q = [item for sublist in X_obs_Q for item in sublist]
X_obs_Z = [item for sublist in X_obs_Z for item in sublist]
X_obs_S = [item for sublist in X_obs_S for item in sublist]
X_obs_A0 = [item for sublist in X_obs_A0 for item in sublist]
X_obs_A = [item for sublist in X_obs_A for item in sublist]
X_obs_B = [item for sublist in X_obs_B for item in sublist]
X_obs_w = [item for sublist in X_obs_w for item in sublist]
X_obs_a = [item for sublist in X_obs_a for item in sublist]
X_obs_delta_S = [item for sublist in X_obs_delta_S for item in sublist]

X_obs = np.array(X_obs)
X_obs_Q = np.array(X_obs_Q)
X_obs_Z = np.array(X_obs_Z)
X_obs_S = np.array(X_obs_S)
X_obs_A0 = np.array(X_obs_A0)
X_obs_A = np.array(X_obs_A)
X_obs_B = np.array(X_obs_B)
X_obs_w = np.array(X_obs_w)
X_obs_a = np.array(X_obs_a)
X_obs_delta_S = np.array(X_obs_delta_S)

# 输出查看结果
print("X_obs:", X_obs.shape)
print(X_obs_Q.shape)
print(X_obs_Z.shape)
print(X_obs_S.shape)
print(X_obs_A.shape)
print(X_obs_B.shape)
print("X_obs_A0:", X_obs_A0.shape)
print(X_obs_a.shape)
print(X_obs_delta_S.shape)
