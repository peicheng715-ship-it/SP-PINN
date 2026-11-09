"""不缩放无输入adapt"""
import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
import warnings
import data7
import test
warnings.filterwarnings("ignore")

from Adaptno import SVE

import data3
import data4
import data5


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config.gpu_options.allow_growth = True

sess0 = tf.InteractiveSession(config=config)


if __name__ == "__main__":
    x_total = data3.x_total
    t_total = data4.t_total

    layers = [2, 256, 512, 5]

    useObs = True

    # 一级归一化后的组合（x，t）
    X_star = [[x_total[i][0], t_total[i][0]] for i in range(len(x_total))]
    X_star = np.array(X_star)
    print("X_star：\n", X_star)
    X_f_train = X_star
    X_u_BC = data5.X_u_BC
    X_h_BC = data5.X_h_BC
    X_Z_BC = data5.X_Z_BC
    X_u_obs = test.X_u_obs  # 已填充
    X_h_obs = test.X_u_obs  # 已填充
    Q_BC = data5.Q_BC
    S_BC = data5.S_BC
    Z_BC = data5.Z_BC
    H_BC = data5.H_BC
    # Q_obs = test.Q_obs  # 已填充
    Q_obs = test.Q_obs
    Z_obs = test.Z_obs
    H_obs = test.H_obs
    U_obs = test.U_obs
    S_obs = test.S_obs
    A0_obs = test.A0_obs
    b = 1
    # 缩放后的值 S没有缩放 取的S0 = 1
    Q_star = data5.Q_star
    Z_star = data5.Z_star
    H_star = data5.H_star
    U_star = data5.U_star
    S_star = data5.S_star
    A0_star = data5.A0_star
    n_star = data3.n

    B = data3.B
    A = data3.A
    Jf = data4.Jf
    J1 = 1  # 暂不考虑
    hc = data3.hc
    w = data3.w
    a = data3.a
    S_1 = data4.S_1
    pm = data4.pm
    Delta_p = 1650
    ps = 2650
    delta_S = data4.delta_S
    delta_S1 = data4.delta_S1
    P = 1400
    A_x = data5.A_x

    # 获取   输入数据  中每列的最小值和最大值
    x_total = np.array(x_total)
    t_total = np.array(t_total)
    A = np.array(A)
    B = np.array(B)
    S_1 = np.array(S_1)
    Jf = np.array(Jf)
    hc = np.array(hc)
    pm = np.array(pm)
    A_x = np.array(A_x)

    X_contact = [[x_total[i],t_total[i]] for i in range(len(x_total))]

    X_f_np = np.array(X_contact)


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

    # 边界BC入口参数
    x_BC_A = data7.x_BC_A
    x_BC_B = data7.x_BC_B
    x_BC_Jf = data7.x_BC_Jf
    x_BC_hc = data7.x_BC_hc
    x_BC_w = data7.x_BC_w
    x_BC_a = data7.x_BC_a
    x_BC_S_1 = data7.x_BC_S_1
    x_BC_pm = data7.x_BC_pm
    x_BC_delta_S = data7.x_BC_delta_S
    x_BC_delta_S1 = data7.x_BC_delta_S1
    x_BC_Ax = data7.x_BC_Ax
    # 边界BC出口参数
    x_Z_BC_A = data7.x_Z_BC_A
    x_Z_BC_B = data7.x_Z_BC_B
    x_Z_BC_Jf = data7.x_Z_BC_Jf
    x_Z_BC_hc = data7.x_Z_BC_hc
    x_Z_BC_w = data7.x_Z_BC_w
    x_Z_BC_a = data7.x_Z_BC_a
    x_Z_BC_S_1 = data7.x_Z_BC_S_1
    x_Z_BC_pm = data7.x_Z_BC_pm
    x_Z_BC_delta_S = data7.x_Z_BC_delta_S
    x_Z_BC_delta_S1 = data7.x_Z_BC_delta_S1
    x_Z_BC_Ax = data7.x_Z_BC_Ax

    # 观测obs数据输入参数
    x_A_obs = data7.x_A_obs
    x_B_obs = data7.x_B_obs
    x_hc_obs = data7.x_hc_obs
    x_w_obs = data7.x_w_obs
    x_Jf_obs = data7.x_Jf_obs
    x_delta_S1_obs = data7.x_delta_S1_obs
    x_delta_S_obs = data7.x_delta_S_obs
    x_S1_obs = data7.x_S1_obs
    x_a_obs = data7.x_a_obs
    x_Ax_obs = data7.x_Ax_obs
    x_pm_obs = data7.x_pm_obs

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

    # 训练数据输入参数
    x_B = data3.B
    x_A = data3.A
    x_Jf = data4.Jf
    x_hc = data3.hc
    x_w = data3.w
    x_a = data3.a
    x_S_1 = data4.S_1
    x_pm = data4.pm
    x_delta_S = data4.delta_S
    x_delta_S1 = data4.delta_S1
    x_Ax = data5.A_x

    print("Shape of x:", x_total.shape)
    print("Shape of t:", t_total.shape)
    print("Shape of A:", x_A.shape)
    print("Shape of B:", x_B.shape)
    print("Shape of S_1:", x_S_1.shape)
    print("Shape of delta_S1:", x_delta_S1.shape)
    print("Shape of w:", x_w.shape)
    print("Shape of a:", x_a.shape)
    print("Shape of Jf:", x_Jf.shape)
    print("Shape of hc:", x_hc.shape)
    print("Shape of delta_S:", x_delta_S.shape)
    print("Shape of pm:", x_pm.shape)
    print("Shape of Ax:", x_Ax.shape)


    exist_mode = 0
    saved_path = ''
    weight_path = ''
    model2 = SVE(
        X_u_BC, X_h_BC, x_BC_A, x_BC_B, x_BC_Jf, x_BC_hc, x_BC_w, x_BC_a, x_BC_S_1, x_BC_pm, x_BC_delta_S, x_BC_delta_S1, x_BC_Ax,  # 边界入口输入（x，t，参数）
        X_Z_BC, x_Z_BC_A, x_Z_BC_B, x_Z_BC_Jf, x_Z_BC_hc, x_Z_BC_w, x_Z_BC_a, x_Z_BC_S_1, x_Z_BC_pm, x_Z_BC_delta_S, x_Z_BC_delta_S1, x_Z_BC_Ax,  # 边界出口输入（x，t，参数）
        X_u_obs, X_h_obs, x_A_obs, x_B_obs, x_hc_obs, x_Jf_obs, x_delta_S1_obs, x_delta_S_obs, x_w_obs, x_S1_obs, x_a_obs, x_Ax_obs, x_pm_obs, # 观测输入（x，t，参数）
        X_f_train, x_A, x_B, x_S_1, x_delta_S1, x_w, x_a, x_Jf, x_hc, x_delta_S, x_pm, x_Ax, # 方程输入（x，t，参数）
        Q_BC, S_BC, H_BC,  # 边界 真实值
        Q_obs, H_obs, U_obs, S_obs, A0_obs,  # 观测真实值
        layers,
        lb, ub, b,
        X_star, Q_star, H_star, U_star, S_star, A0_star,n_star,  # 方程解的真实值
        B, A, Jf, J1, hc, w, a, S_1, pm, Delta_p, ps, delta_S, delta_S1, P, A_x,
        ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path,
        useObs=True)

    model2.train(40000)
    filepath2 = r'结果/PINN_share_SVE.pickle'
    filepath3 = r'结果/PINN_Q_SVE.pickle'
    filepath4 = r'结果/PINN_Z_SVE.pickle'
    filepath5 = r'结果/PINN_S_SVE.pickle'
    filepath6 = "结果/2024.11.19数据处理"
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

    Z_test = Z_star   # 真实值
    H_test = H_star
    Q_test = Q_star
    S_test = S_star
    U_test = U_star


    # 预测值
    Q_pred2, H_pred2, U_pred2, S_pred2, A0_pred2 = model2.predict(x_test, t_test)
    # 计算误差指标
    Q0 = 1
    S0 = 1
    error_H = np.linalg.norm(H_test - H_pred2, 2) / np.linalg.norm(H_test, 2)
    error_Q = np.linalg.norm(Q_test * Q0 - Q_pred2 * Q0, 2) / np.linalg.norm(Q_test * Q0, 2)
    error_U = np.linalg.norm(U_test - U_pred2, 2) / np.linalg.norm(U_test, 2)
    error_S = np.linalg.norm(S_test / S0 - S_pred2 / S0, 2) / np.linalg.norm(S_test / S0, 2)


    # 创建主数据DataFrame
    data_dict = {
        'x_test': x_test.flatten(),
        't_test': t_test.flatten(),
        'Q_pred2': (Q_pred2 * Q0).flatten(),
        'H_pred2': H_pred2.flatten(),
        'U_pred2': U_pred2.flatten(),
        'S_pred2': (S_pred2 / S0).flatten(),
        'A0_pred2': A0_pred2.flatten()
    }

    df_predictions = pd.DataFrame(data_dict)


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
    nse_H = nse(H_test, H_pred2)
    nse_U = nse(U_test, U_pred2)
    nse_Q = nse(Q_test, Q_pred2)
    nse_S = nse(S_test, S_pred2)  # 如果有S的话

    print(f"水位(Z) NSE: {nse_H:.6f}")
    print(f"流量(Q) NSE: {nse_Q:.6f}")
    print(f"流量(U) NSE: {nse_U:.6f}")
    print(f"流量(S) NSE: {nse_S:.6f}")

    # 创建误差信息DataFrame
    error_info = pd.DataFrame({
        '指标': ['相对误差_Q', '相对误差_H', '相对误差_U', '相对误差_S', 'nse_Q', 'nse_H', 'nse_U', 'nse_S'],
        '值': [error_Q, error_H, error_U, error_S, nse_Q, nse_H, nse_U, nse_S],
    })

    # 确保目录存在
    output_dir = "结果/无输入/不缩放adapt/网络512"
    os.makedirs(output_dir, exist_ok=True)

    output_file_excel = os.path.join(output_dir, "model_predictions.xlsx")

    # 使用ExcelWriter创建多sheet的Excel文件
    with pd.ExcelWriter(output_file_excel, engine='openpyxl') as writer:
        # 写入预测结果
        df_predictions.to_excel(writer, sheet_name='预测结果', index=False)

        # 写入误差统计
        error_info.to_excel(writer, sheet_name='误差统计', index=False)

        # 可以添加统计信息sheet
        stats_df = df_predictions.describe()
        stats_df.to_excel(writer, sheet_name='统计信息')

    print(f'数据已保存至: {os.path.abspath(output_file_excel)}')
    print(f'文件包含 {len(df_predictions)} 行数据')
    print(f'数据形状: {df_predictions.shape}')


