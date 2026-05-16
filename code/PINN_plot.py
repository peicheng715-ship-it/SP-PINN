"""有输入不缩放PINN"""
import pandas as pd
import tensorflow as tf
import numpy as np
import warnings

from matplotlib import pyplot as plt

import data1
import data2

warnings.filterwarnings("ignore")

from PINN import SVE

# 检查是否有 GPU 可用
print("Is GPU available:", tf.test.is_gpu_available())


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

sess0 = tf.InteractiveSession(config=config)

if __name__ == "__main__":
    x_total = data1.x_arr
    t_total = data1.t_arr

    X_star = data1.xt_combined
    layers = [19] + 3 * [1 * 128] + [4]
    shared_layers = [19, 64, 32]  # 共享两层
    head1_layers = [32, 128, 1]  # Q 的头
    head2_layers = [32, 128, 1]  # Z 的头
    head3_layers = [32, 2]  # S, A0 的头


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

    X_contact = [[x_total[i], t_total[i], A[i], B[i], w[i], *a[i],
                   *delta_S[i]] for i in range(len(x_total))]
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
        shared_layers, head1_layers, head2_layers, head3_layers, layers,
        lb, ub,
        X_star, Q_star, Z_star, S_star, A0_star,n_star, # 方程解的真实值
        B, A, w, a, S_1, P, delta_S, n, A_x,
        ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path,
        useObs=True)

    model2.train(4000)
    filepath1 = r'结果/插值/PINN_SVE.pickle'
    model2.save_NN(filepath1)

    file_path = r"data/PINN_interpolated_dataset.xlsx"

    df = pd.read_excel(file_path, sheet_name="test_only")

    print("测试数据读取成功:", df.shape)

    # ============================================================
    # 2️⃣ 构造输入（与你的 predict 完全一致）
    # ============================================================

    # 坐标
    x_test = df["x"].values.reshape(-1, 1)
    t_test = df["t"].values.reshape(-1, 1)

    # 参数
    A_test = df["A_interp"].values.reshape(-1, 1)
    B_test = df["B_interp"].values.reshape(-1, 1)
    w_test = df["W_interp"].values.reshape(-1, 1)

    # a (7维)
    a_test = df[["a1", "a2", "a3", "a4", "a5", "a6", "a7"]].values

    # delta_S (7维)
    delta_S_test = df[["delta_S1", "delta_S2", "delta_S3",
                       "delta_S4", "delta_S5", "delta_S6", "delta_S7"]].values

    # ============================================================
    # 3️⃣ 真实值（用于评估）
    # ============================================================
    Q_test = df["Q_interp"].values.reshape(-1, 1)
    Z_test = df["H_interp"].values.reshape(-1, 1)
    S_test = df["S_interp"].values.reshape(-1, 1)

    # ============================================================
    # 4️⃣ 模型预测
    # ============================================================
    Q_pred, Z_pred, S_pred, A0_pred = model2.predict(
        x_test, t_test, A_test, B_test, w_test, a_test, delta_S_test
    )

    # ============================================================
    # 5️⃣ 误差计算
    # ============================================================

    Q0 = 500
    S0 = 10

    # 相对误差（L2）
    error_Q = np.linalg.norm(Q_test * Q0 - Q_pred * Q0, 2) / np.linalg.norm(Q_test * Q0, 2)
    error_Z = np.linalg.norm(Z_test - Z_pred, 2) / np.linalg.norm(Z_test, 2)
    error_S = np.linalg.norm(S_test / 1000 - S_pred / S0, 2) / np.linalg.norm(S_test / 1000, 2)

    print(f"Error Q: {error_Q:e}")
    print(f"Error Z: {error_Z:e}")
    print(f"Error S: {error_S:e}")


    # ============================================================
    # 6️⃣ NSE 指标
    # ============================================================
    def nse(obs, pred):
        obs = obs.flatten()
        pred = pred.flatten()

        mask = ~(np.isnan(obs) | np.isnan(pred))
        obs = obs[mask]
        pred = pred[mask]

        return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)


    nse_Q = nse(Q_test, Q_pred)
    nse_Z = nse(Z_test, Z_pred)
    nse_S = nse(S_test/ 1000, S_pred/ S0)

    print(f"NSE Q: {nse_Q:.6f}")
    print(f"NSE Z: {nse_Z:.6f}")
    print(f"NSE S: {nse_S:.6f}")

    # ============================================================
    # 7️⃣ 保存预测结果
    # ============================================================

    output_dir = r"结果/插值"
    os.makedirs(output_dir, exist_ok=True)

    df_out = pd.DataFrame({
        "x": x_test.flatten(),
        "t": t_test.flatten(),

        "Q_true": (Q_test * Q0).flatten(),
        "Q_pred": (Q_pred * Q0).flatten(),

        "Z_true": Z_test.flatten(),
        "Z_pred": Z_pred.flatten(),

        "S_true": (S_test / 1000).flatten(),
        "S_pred": (S_pred / S0).flatten(),

        "A0_pred": A0_pred.flatten()
    })

    df_error = pd.DataFrame({
        "指标": ["Error_Q", "Error_Z", "Error_S", "NSE_Q", "NSE_Z", "NSE_S"],
        "值": [error_Q, error_Z, error_S, nse_Q, nse_Z, nse_S]
    })

    save_path = os.path.join(output_dir, "插值测试结果.xlsx")

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="预测结果", index=False)
        df_error.to_excel(writer, sheet_name="误差指标", index=False)

    print("测试完成，结果保存至:", save_path)


