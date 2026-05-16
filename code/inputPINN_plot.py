"""全连接神经网络 + 输入变为（x，t）"""
import tensorflow as tf
import numpy as np
from datetime import datetime
import warnings

import data1
import data2

warnings.filterwarnings("ignore")

from inputPINN import SVE

# 检查是否有 GPU 可用
print("Is GPU available:", tf.test.is_gpu_available())


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

sess0 = tf.InteractiveSession(config=config)



def time_convert(intime):
    """
    function to convert the time from string to datetime
    """
    Nt = intime.shape[0]
    outtime = []
    for t in range(Nt):
        timestr = intime[t].decode('utf-8')
        outtime.append(datetime.strptime(timestr, '%d%b%Y %H:%M:%S'))
    return outtime


if __name__ == "__main__":
    x_total = data1.x_arr
    t_total = data1.t_arr

    X_star = data1.xt_combined
    # Domain bounds,获取矩阵或数组 X_star 中每列的最小值和最大值
    #lb = X_star.min(0)
    #ub = X_star.max(0)

    layers = [2] + 3 * [1 * 128] + [4]

    useObs = True

    X_f_train = X_star  # 改
    print("Min value:", np.min(X_f_train))
    print("Max value:", np.max(X_f_train))
    print("Any NaN?", np.isnan(X_f_train).any())
    print("Any Inf?", np.isinf(X_f_train).any())
    X_u_BC = data2.BC_X  # 改
    X_h_BC = data2.BC_X
    X_Z_BC = data2.BC_Z_X  # 下游边界输入X
    X_u_obs = data2.X_obs
    X_h_obs = data2.X_obs
    Q_BC = data2.BC_Q  # 改
    Z_BC = data2.BC_Z
    Q_obs = data2.X_obs_Q  # 已填充
    Z_obs = data2.X_obs_Z
    S_obs = data2.X_obs_S
    A0_obs = data2.X_obs_A0
    Q_star = data1.Q  # 改
    Z_star = data1.H  # 改
    S_star = data1.S_data  # 改
    A0_star = data1.A0_data
    n_star = data1.n
    # Q_x = data6.Q_x
    # Z_x = data6.Z_x
    # Q_t = data6.Q_t
    # Z_t = data6.Z_t

    B = data1.B
    A = data1.A
    print("Any NaN?B", np.isnan(B).any())
    print("Any Inf?B", np.isinf(B).any())
    print("Any NaN?A", np.isnan(A).any())
    print("Any Inf?A", np.isinf(A).any())
    w = data1.W  # 分组浑水沉速
    a = data1.a  # 分组恢复饱和系数
    S_1 = data1.S_1  # 挟沙力
    P = data1.P  # 床沙干密度
    delta_S = data1.delta_S_all  # 分组悬沙级配
    n = data1.n  # 曼宁系数 n
    A_x = data1.A_x

    X_contact = [[x_total[i], t_total[i]] for i in range(len(x_total))]
    # X_contact = [[x_total[i], t_total[i], A[i], B[i]] for i in range(len(x_total))]
    print("len(X_contact)", len(X_contact))
    X_contact_np = np.array(X_contact)


    X_f_np = np.array(X_contact)
    print(X_f_np.shape)
    lb = np.array([item.item() if isinstance(item, np.ndarray) else item for item in X_f_np.min(0)])
    ub = np.array([item.item() if isinstance(item, np.ndarray) else item for item in X_f_np.max(0)])
    # 检测常量特征
    constant_indices = np.where((ub - lb) == 0)[0]

    # 修改常量特征的 ub 和 lb 值，确保 (ub - lb) 不为 0
    if constant_indices.size > 0:
        ub[constant_indices] += 1e-8  # 或者直接设置 ub = lb + ε

    c = ub - lb
    print("lb=", lb)
    print("ub=", ub)
    print("c=", c)

    x_f_np = x_total
    lb2 = np.array([item.item() if isinstance(item, np.ndarray) else item for item in x_f_np.min(0)])
    ub2 = np.array([item.item() if isinstance(item, np.ndarray) else item for item in x_f_np.max(0)])
    print("lb2=", lb2)
    print("ub2=", ub2)


    # 边界BC入口参数  已改
    x_BC_A = data2.x_BC_A
    x_BC_B = data2.x_BC_B
    x_BC_w = data2.x_BC_w
    x_BC_a = data2.x_BC_a
    x_BC_delta_S = data2.x_BC_delta_S

    # 边界BC出口参数
    x_Z_BC_A = data2.x_BC_Z_A
    x_Z_BC_B = data2.x_BC_Z_B
    x_Z_BC_w = data2.x_BC_Z_w
    x_Z_BC_a = data2.x_BC_Z_a
    x_Z_BC_delta_S = data2.x_BC_Z_delta_S

    # 观测obs数据输入参数
    x_A_obs = data2.X_obs_A
    x_B_obs = data2.X_obs_B
    x_w_obs = data2.X_obs_w
    x_delta_S_obs = data2.X_obs_delta_S
    x_a_obs = data2.X_obs_a

    x_B = data1.B
    x_A = data1.A
    x_w = data1.W
    x_a = data1.a
    x_delta_S = data1.delta_S_all

    print("Shape of x:", x_total.shape)
    print("Shape of t:", t_total.shape)
    print("Shape of A:", x_A.shape)
    print("Shape of B:", x_B.shape)
    print("Shape of w:", x_w.shape)
    print("Shape of a:", x_a.shape)
    print("Shape of delta_S:", x_delta_S.shape)

    exist_mode = 0
    exist_mode1 = 0
    exist_mode2 = 0
    saved_path = ''
    weight_path = ''
    model2 = SVE(
        X_u_BC, X_h_BC, x_BC_A, x_BC_B, x_BC_w, x_BC_a, x_BC_delta_S,  # 边界入口输入（x，t，参数）
        X_Z_BC, x_Z_BC_A, x_Z_BC_B, x_Z_BC_w, x_Z_BC_a, x_Z_BC_delta_S,  # 边界出口输入（x，t，参数）
        X_u_obs, X_h_obs, x_A_obs, x_B_obs, x_delta_S_obs, x_w_obs, x_a_obs,  # 观测输入（x，t，参数）
        X_f_train, x_A, x_B, x_w, x_a, x_delta_S,  # 方程输入（x，t，参数）
        Q_BC, Z_BC,  # 边界 真实值
        Q_obs, Z_obs, S_obs, A0_obs,  # 观测真实值
        layers,
        lb, ub,
        X_star, Q_star, Z_star, S_star, A0_star, n_star, # 方程解的真实值
        B, A, w, a, S_1, P, delta_S, n, A_x,
        ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path,
        useObs=True)

    model2.train(7000)
    # filepath2 = r'change n/结果/PINN_shared_SVE.pickle'
    # filepath3 = r'change n/结果/PINN_Q_SVE.pickle'
    # filepath4 = r'change n/结果/PINN_Z_SVE.pickle'
    # filepath5 = r'change n/结果/PINN_S_SVE.pickle'
    # filepath6 = r'change n/结果/weights.out'
    # model2.save_weight(filepath1)

    filepath1 = r'结果/weights.out'
    # model2.save_weight(filepath1)
    # model2.save_share_NN(filepath2, filepath6)
    # model2.save_Q_NN(filepath3, filepath6)
    # model2.save_Z_NN(filepath4, filepath6)
    # model2.save_S_NN(filepath5, filepath6)


    # Test data
    X_test = X_star
    x_test = X_test[:, 0:1]
    t_test = X_test[:, 1:2]

    Z_test = Z_star  # 真实值
    Q_test = Q_star
    S_test = S_star
    A0_test = A0_star
    # 预测值
    Q_pred2, Z_pred2,S_pred2,A0_pred2 = model2.predict(x_test, t_test)
    # Q_pred2, Z_pred2 = model2.predict1(x_test, t_test)
    # n_pred = model2.predict2(x_test)

    error_Z = np.linalg.norm(Z_test - Z_pred2, 2) / np.linalg.norm(Z_test, 2)
    # error_S = np.linalg.norm(S_test - S_pred2, 2) / np.linalg.norm(S_test, 2)
    error_Q2 = np.linalg.norm(Q_test - Q_pred2, 2) / np.linalg.norm(Q_test, 2)
    # error_A0 = np.linalg.norm(A0_test - A0_pred2, 2) / np.linalg.norm(A0_test, 2)
    # error_n = np.linalg.norm(n_star - n_pred, 2) / np.linalg.norm(Z_test, 2)

    print('Error : %e' % (error_Q2 + error_Z))

    rmse_Q = np.sqrt(((Q_test - Q_pred2) ** 2).mean())
    print('RMSE Q: %.3f m' % rmse_Q)

    # 记录结果
    data = np.column_stack((x_test, t_test, Q_pred2, Z_pred2,S_pred2, A0_pred2))
    # data = np.column_stack((x_test, t_test, Q_pred2, Z_pred2))

    # 确保目录存在
    output_file = "结果/input/model_predictions.txt"

    # 写入数据到 txt 文件
    with open(output_file, 'w') as f:
        # 写入表头
        f.write("x_test, t_test, Q_pred2, Z_pred2, S_pred2, A0_pred2\n")
        # f.write("x_test, t_test, Q_pred2, Z_pred2\n")
        # 写入每一行数据
        np.savetxt(f, data, fmt="%.6f", delimiter=", ")

    print(f'Data saved to {os.path.abspath(output_file)}')

    def nse(observed, simulated):
        """
        计算Nash-Sutcliffe效率系数
        NSE = 1 - ∑(观测值-模拟值)² / ∑(观测值-观测均值)²
        """
        observed = np.array(observed).flatten()
        simulated = np.array(simulated).flatten()

        # 移除NaN值
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        observed = observed[mask]
        simulated = simulated[mask]

        numerator = np.sum((observed - simulated) ** 2)
        denominator = np.sum((observed - np.mean(observed)) ** 2)

        nse_value = 1 - numerator / denominator
        return nse_value


    # 计算各变量的NSE
    nse_Z = nse(Z_test, Z_pred2)
    nse_Q = nse(Q_test, Q_pred2)
    nse_S = nse(S_test, S_pred2)  # 如果有S的话

    print(f"水位(Z) NSE: {nse_Z:.6f}")
    print(f"流量(Q) NSE: {nse_Q:.6f}")
    print(f"流量(S) NSE: {nse_S:.6f}")


