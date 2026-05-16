"""损失加权方法 自适应"""
import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
import warnings
import data1
import data2
import os
warnings.filterwarnings("ignore")

from scaleAdaptno import SVE

# 检查是否有 GPU 可用
print("Is GPU available:", tf.test.is_gpu_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

sess0 = tf.InteractiveSession(config=config)



if __name__ == "__main__":
    x_total = data1.x_arr
    t_total = data1.t_arr

    X_star = data1.xt_combined

    layers = [2] + 3 * [1 * 128] + [4]
    # shared_layers = [19, 128]  # 共享两层
    # head1_layers = [128, 128, 128, 1]  # Q 的头
    # head2_layers = [128, 128, 128, 1]  # Z 的头
    # head3_layers = [128, 128, 128, 2]  # S, A0 的头
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

    model2.train(2520)
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
    Q_pred2, Z_pred2, S_pred2, A0_pred2 = model2.predict(x_test, t_test)

    # 计算误差指标
    Q0 = 500
    S0 = 10
    error_Z = np.linalg.norm(Z_test - Z_pred2, 2) / np.linalg.norm(Z_test, 2)
    error_Q2 = np.linalg.norm(Q_test * Q0 - Q_pred2 * Q0, 2) / np.linalg.norm(Q_test * Q0, 2)
    error_S = np.linalg.norm(S_test / S0 - S_pred2 / S0, 2) / np.linalg.norm(S_test / S0, 2)

    print(f'ErrorQ: {error_Q2:e}')
    print(f'ErrorZ: {error_Z:e}')
    print(f'ErrorS: {error_S:e}')

    # 创建主数据DataFrame
    data_dict = {
        'x_test': x_test.flatten(),
        't_test': t_test.flatten(),
        'Q_pred2': (Q_pred2 * Q0).flatten(),
        'Z_pred2': Z_pred2.flatten(),
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
    nse_Z = nse(Z_test, Z_pred2)
    nse_Q = nse(Q_test, Q_pred2)
    nse_S = nse(S_test, S_pred2)  # 如果有S的话

    print(f"水位(Z) NSE: {nse_Z:.6f}")
    print(f"流量(Q) NSE: {nse_Q:.6f}")
    print(f"流量(S) NSE: {nse_S:.6f}")

    # 创建误差信息DataFrame
    error_info = pd.DataFrame({
        '指标': ['相对误差_Q', '相对误差_Z', '相对误差_S', 'nse_Z', 'nse_Q', 'nse_S'],
        '值': [error_Z, error_Q2, error_S, nse_Z, nse_Q, nse_S],
    })

    # 确保目录存在
    output_dir = "结果/实验Z/无输入/缩放500/缩放adapt/最佳轮次"
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


