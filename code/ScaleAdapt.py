import numpy as np
from datetime import datetime
import time

import pandas as pd
import tensorflow as tf
print(tf.__version__)
import pickle
import os

import warnings
warnings.filterwarnings("ignore")
np.random.seed(1234)
tf.set_random_seed(1234)


class SVE:
    DTYPE = tf.float32

    def __init__(self,
                 X_u_BC, X_h_BC, x_BC_A, x_BC_B, x_BC_w, x_BC_a, x_BC_delta_S,  # 边界入口输入（x，t，参数）
                 X_Z_BC, x_Z_BC_A, x_Z_BC_B, x_Z_BC_w, x_Z_BC_a, x_Z_BC_delta_S,  # 边界出口输入（x，t，参数）
                 X_u_obs, X_h_obs, x_A_obs, x_B_obs, x_delta_S_obs, x_w_obs, x_a_obs,  # 观测输入（x，t，参数）
                 X_f, x_A, x_B, x_w, x_a, x_delta_S,  # 方程输入（x，t，参数）
                 Q_BC, Z_BC,  # 边界 真实值
                 Q_obs, Z_obs, S_obs, A0_obs,  # 观测真实值
                 layers,
                 lb, ub,
                 X_star, Q_star, Z_star, S_star, A0_star, n_star,  # 方程解的真实值
                 B, A, w, a, S_1, P, delta_S, n, A_x,  # 损失函数使用
                 lr=2e-4,
                 ExistModel=0, uhDir='', wDir='', useObs=True):

        self.count = 0
        self.Q0 = tf.constant(1, dtype=self.DTYPE)  # 相对密度1650

        # 方程参数  w,a,s_1可能还需要按分组修改
        self.B = B  # 水面宽度
        self.A = A  # 断面过水面积
        # w和a待定
        self.w = w
        self.a = a  # a同上

        self.S_1 = S_1  # 挟沙力
        self.g = tf.constant(9.81, dtype=self.DTYPE)
        self.alpha_f = tf.constant(1, dtype=self.DTYPE)  # 后续修改值
        self.delta_S = delta_S  # 悬沙级配
        self.P = tf.constant(P, dtype=self.DTYPE)  # 床沙干密度 1400
        self.A_x = A_x   # 导数值
        self.lb = lb
        self.ub = ub
        self.useObs = useObs

        # test data  # 五个输出的真实值已改
        self.X_star = X_star
        self.Q_star = Q_star
        self.Z_star = Z_star
        self.S_star = S_star
        self.A0_star = A0_star
        self.n_star = n_star

        # Adaptive re-weighting constant
        self.beta = 0.9
        self.adaptive_constant_bcs_Q_val = np.array(1.0)  # 已改
        self.adaptive_constant_bcs_S_val = np.array(1.0)  # 已改
        self.adaptive_constant_bcs_Z_val = np.array(1.0)  # 新增
        # self.adaptive_constant_ics_h_val = np.array(1.0) # 删除
        self.adaptive_constant_obs_Q_val = np.array(1.0)  # 新增
        self.adaptive_constant_obs_Z_val = np.array(1.0)
        self.adaptive_constant_obs_U_val = np.array(1.0)
        self.adaptive_constant_obs_S_val = np.array(1.0)
        self.adaptive_constant_obs_A0_val = np.array(1.0)

        # 边界BC入口出口输入（x，t）
        self.x_u_BC = X_u_BC[:, 0:1]     # 实际使用
        self.t_u_BC = X_u_BC[:, 1:2]     # 实际使用
        self.x_h_BC = X_h_BC[:, 0:1]     # 没用
        self.t_h_BC = X_h_BC[:, 1:2]     # 没用
        self.x_Z_BC = X_Z_BC[:, 0:1]  # 新增
        self.t_Z_BC = X_Z_BC[:, 1:2]  # 新增
        # 边界BC入口输入（参数）
        self.x_BC_A = x_BC_A
        self.x_BC_B = x_BC_B
        self.x_BC_w = x_BC_w
        self.x_BC_a = x_BC_a
        self.x_BC_delta_S = x_BC_delta_S

        # 边界BC出口输入（参数）
        self.x_Z_BC_A = x_Z_BC_A
        self.x_Z_BC_B = x_Z_BC_B
        self.x_Z_BC_w = x_Z_BC_w
        self.x_Z_BC_a = x_Z_BC_a
        self.x_Z_BC_delta_S = x_Z_BC_delta_S

        # 观测数据输入（x,t)
        self.x_u_obs = X_u_obs[:, 0:1]
        self.t_u_obs = X_u_obs[:, 1:2]
        self.x_h_obs = X_h_obs[:, 0:1]
        self.t_h_obs = X_h_obs[:, 1:2]
        # 观测数据输入（参数）
        self.x_A_obs = x_A_obs
        self.x_B_obs = x_B_obs
        self.x_delta_S_obs = x_delta_S_obs
        self.x_w_obs = x_w_obs
        self.x_a_obs = x_a_obs

        # 网络输入（x,t,所有参数）
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.x_A = x_A
        self.x_B = x_B
        self.x_w = x_w
        self.x_a = x_a
        self.x_delta_S = x_delta_S
        self.x_Ax = A_x

        self.Q_BC = Q_BC  # 已改
        self.Z_BC = Z_BC  # 新增
        self.Q_obs = Q_obs
        self.Z_obs = Z_obs
        self.S_obs = S_obs
        self.A0_obs = A0_obs

        # layers
        # self.layers = layers
        # self.share_layers = shared_layers
        # self.head1_layers = head1_layers
        # self.head2_layers = head2_layers
        # self.head3_layers = head3_layers
        self.layers = layers
        # initialize NN
        if ExistModel == 0:
            # self.weights_shared, self.biases_shared = self.initialize_NN(shared_layers, scope="shared")
            # self.weights_head1, self.biases_head1 = self.initialize_NN(head1_layers, scope="head1")
            # self.weights_head2, self.biases_head2 = self.initialize_NN(head2_layers, scope="head2")
            # self.weights_head3, self.biases_head3 = self.initialize_NN(head3_layers, scope="head3")
            self.weights, self.biases = self.initialize_NN(layers, scope="NN")
        # else:
        #     print("Loading uh NN ...")
        #     self.weights, self.biases = \
        #         self.load_NN(uhDir, layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        # placeholders for data on velocities (inside the domain)
        # 训练数据输入占位符（x,t,参数）
        self.x_u_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_u_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])
        self.x_A_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_A.shape[1]])
        self.x_B_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_B.shape[1]])
        self.x_w_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_w.shape[1]])
        self.x_a_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_a.shape[1]])
        self.x_delta_S_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_delta_S.shape[1]])
        self.x_Ax_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_Ax.shape[1]])
        # 暂未使用
        self.x_h_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_h_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])

        # 边界BC入口输入占位符（x,t）    BC入口及出口真实值（Q,S,Z）
        self.x_u_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_u_BC.shape[1]])
        self.t_u_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_u_BC.shape[1]])
        self.Q_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.Q_BC.shape[1]])  # 已改
        self.x_h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_h_BC.shape[1]])
        self.t_h_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_h_BC.shape[1]])
        self.x_Z_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_Z_BC.shape[1]])  # 新增输入
        self.t_Z_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_Z_BC.shape[1]])  # 新增输入
        self.Z_BC_tf = tf.placeholder(self.DTYPE, shape=[None, self.Z_BC.shape[1]])  # 新增  输出
        # 边界BC入口输入占位符（参数）
        self.x_BC_A_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_BC_A.shape[1]])
        self.x_BC_B_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_BC_B.shape[1]])
        self.x_BC_w_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_BC_w.shape[1]])
        self.x_BC_a_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_BC_a.shape[1]])
        self.x_BC_delta_S_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_BC_delta_S.shape[1]])
        # 边界BC出口输入占位符（参数）
        self.x_Z_BC_A_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_Z_BC_A.shape[1]])
        self.x_Z_BC_B_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_Z_BC_B.shape[1]])
        # print("报错之前打印：")
        # print("Shape of self.x_Z_BC_Jf:", self.x_Z_BC_Jf.shape)
        self.x_Z_BC_w_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_Z_BC_w.shape[1]])
        self.x_Z_BC_a_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_Z_BC_a.shape[1]])
        self.x_Z_BC_delta_S_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_Z_BC_delta_S.shape[1]])

        # 观测obs数据输入（x,t,Q,Z,U,S,A0)
        self.x_u_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_u_obs.shape[1]])
        self.t_u_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_u_obs.shape[1]])
        self.Q_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.Q_obs.shape[1]])  # 新增
        self.x_h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_h_obs.shape[1]])
        self.t_h_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_h_obs.shape[1]])
        self.Z_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.Z_obs.shape[1]])  # 新增
        self.S_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.S_obs.shape[1]])  # 新增
        self.A0_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.A0_obs.shape[1]])  # 新增
        # 观测obs数据输入（参数）
        self.x_A_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_A_obs.shape[1]])
        self.x_B_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_B_obs.shape[1]])
        self.x_delta_S_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_delta_S_obs.shape[1]])
        self.x_w_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_w_obs.shape[1]])
        self.x_a_obs_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_a_obs.shape[1]])


        self.x_f_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])

        self.dummy_tf = tf.placeholder(self.DTYPE, shape=(None, layers[-1]))  # dummy variable for fwd_gradients

        if ExistModel == 0:
            self.adaptive_constant_bcs_Q_tf = tf.placeholder(tf.float32,
                                                             shape=self.adaptive_constant_bcs_Q_val.shape)  # 已改
            self.adaptive_constant_bcs_Z_tf = tf.placeholder(tf.float32,
                                                             shape=self.adaptive_constant_bcs_Z_val.shape)  # 新增BC
            # self.adaptive_constant_ics_h_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_ics_h_val.shape)
            self.adaptive_constant_obs_Q_tf = tf.placeholder(tf.float32,
                                                             shape=self.adaptive_constant_obs_Q_val.shape)  # 新增
            self.adaptive_constant_obs_Z_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_obs_Z_val.shape)
            self.adaptive_constant_obs_S_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_obs_S_val.shape)
            self.adaptive_constant_obs_A0_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_obs_A0_val.shape)
        elif ExistModel in [1, 2]:  # 加载先前训练好的自适应权重
            print('Loading adaptive weights ...')
            if self.useObs:  # BC已改 IC已删 obs已增
                self.adaptive_constant_bcs_Q_tf, \
                self.adaptive_constant_bcs_S_tf, \
                self.adaptive_constant_bcs_Z_tf, \
                self.adaptive_constant_obs_Q_tf, \
                self.adaptive_constant_obs_Z_tf, \
                self.adaptive_constant_obs_U_tf, \
                self.adaptive_constant_obs_S_tf, \
                self.adaptive_constant_obs_A0_tf = self.load_weight(wDir)
                print(
                    "constant_bcs_Q_val: {:.3f}, constant_bcs_S_val: {:.3f}, constant_bcs_Z_val: {:.3f},  "
                    "constant_obs_Q_val: {:.3f}, constant_obs_Z_val: {:.3f}".format(
                        self.adaptive_constant_bcs_Q_tf, self.adaptive_constant_bcs_S_tf,
                        self.adaptive_constant_bcs_Z_tf,  # self.adaptive_constant_ics_h_tf,
                        self.adaptive_constant_obs_Q_tf, self.adaptive_constant_obs_Z_tf))
            else:  # BC已改 IC已删
                self.adaptive_constant_bcs_Q_tf, \
                self.adaptive_constant_bcs_S_tf, \
                self.adaptive_constant_bcs_Z_tf = self.load_weight(wDir)
                print("constant_bcs_Q_val: {:.3f}, constant_bcs_S_val: {:.3f},constant_bcs_Z_val: {:.3f}, ".format(
                    self.adaptive_constant_bcs_Q_tf, self.adaptive_constant_bcs_S_tf,
                    self.adaptive_constant_bcs_Z_tf))

                # physics informed neural networks
                # 训练数据输出预测值
        self.Q_pred, self.Z_pred, self.S_pred, self.A0_pred = self.net_uh(self.x_u_tf, self.t_u_tf,
                                                                              self.x_A_tf,
                                                                              self.x_B_tf, self.x_w_tf,
                                                                              self.x_a_tf, self.x_delta_S_tf)
        # 边界BC预测值
        self.Q_BC_pred, x, y, z = self.net_uh(self.x_u_BC_tf, self.t_u_BC_tf,
                                                  self.x_BC_A_tf,
                                                  self.x_BC_B_tf, self.x_BC_w_tf,
                                                  self.x_BC_a_tf, self.x_BC_delta_S_tf)

        x, self.Z_BC_pred, y, z = self.net_uh(self.x_Z_BC_tf, self.t_Z_BC_tf,
                                                  self.x_Z_BC_A_tf,
                                                  self.x_Z_BC_B_tf, self.x_Z_BC_w_tf,
                                                  self.x_Z_BC_a_tf, self.x_Z_BC_delta_S_tf)  # 新增预测BC_z
        # 使用useObs 观测值obs预测值
        if self.useObs:
            self.Q_obs_pred, self.Z_obs_pred, self.S_obs_pred, self.A0_obs_pred = self.net_uh(
                    self.x_u_obs_tf, self.t_u_obs_tf,
                    self.x_A_obs_tf, self.x_B_obs_tf,
                    self.x_w_obs_tf, self.x_a_obs_tf,
                    self.x_delta_S_obs_tf)
        # 方程项预测值
        self.eq1_pred, self.eq2_pred, self.eq3_pred, self.eq4_pred = self.net_f(self.x_f_tf, self.t_f_tf,
                                                                                    self.x_A_tf,
                                                                                    self.x_B_tf,
                                                                                    self.x_w_tf,
                                                                                    self.x_a_tf,
                                                                                    self.x_delta_S_tf)

        # loss pde损失已改，BC已改
        self.loss_f_c = tf.reduce_mean(tf.square(self.eq1_pred))  # 连续方程损失
        self.loss_f_m = tf.reduce_mean(tf.square(self.eq2_pred))  # 动量方程损失
        self.loss_f_s = tf.reduce_mean(tf.square(self.eq3_pred))  # 泥沙方程损失
        self.loss_f_b = tf.reduce_mean(tf.square(self.eq4_pred))  # 河床方程损失
        self.loss_f = self.loss_f_c + self.loss_f_m + self.loss_f_s + self.loss_f_b

        self.loss_BC_Q = tf.reduce_mean(tf.square(self.Q_BC_tf - self.Q_BC_pred))  # 已改
        self.loss_BC_Z = tf.reduce_mean(tf.square(self.Z_BC_tf - self.Z_BC_pred))  # 新增
        self.loss_BCs = self.adaptive_constant_bcs_Q_tf * self.loss_BC_Q \
                            + self.adaptive_constant_bcs_Z_tf * self.loss_BC_Z  # 已改 新增

        # 损失函数删掉了IC
        self.loss = self.loss_f + self.loss_BCs

        if self.useObs:  # 新增
            self.loss_obs_Q = tf.reduce_mean(tf.square(self.Q_obs_tf - self.Q_obs_pred))
            self.loss_obs_Z = tf.reduce_mean(tf.square(self.Z_obs_tf - self.Z_obs_pred))
            self.loss_obs_S = tf.reduce_mean(tf.square(self.S_obs_tf - self.S_obs_pred))
            self.loss_obs_A0 = tf.reduce_mean(tf.square(self.A0_obs_tf - self.A0_obs_pred))
            self.loss_obs = self.adaptive_constant_obs_Q_tf * self.loss_obs_Q \
                                + self.adaptive_constant_obs_Z_tf * self.loss_obs_Z \
                                + self.adaptive_constant_obs_S_tf * self.loss_obs_S \
                                + self.adaptive_constant_obs_A0_tf * self.loss_obs_A0
            self.loss += self.loss_obs


        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 50000,
                                                                             'maxfun': 50000,
                                                                             'maxcor': 50,
                                                                             'maxls': 50,
                                                                             'ftol': 1.0e-10,
                                                                             'gtol': 0.000001})
        warmup_steps = 5000  # warmup 的步数
        total_steps = 100000  # 总的训练步数
        initial_lr = lr  # 最终的学习率

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = self.get_lr_with_warmup(self.global_step, lr, warmup_steps, total_steps)
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # 2. 计算梯度（未裁剪）
        gradients, variables = zip(*self.optimizer_Adam.compute_gradients(self.loss))
        # 3. 梯度裁剪（限制梯度范围，防止爆炸）
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)  # 可调整 clip_norm
        # 4. 应用裁剪后的梯度
        self.train_op_Adam = self.optimizer_Adam.apply_gradients(
                zip(clipped_gradients, variables),
                global_step=self.global_step
        )


        self.loss_f_c_log = []
        self.loss_f_m_log = []
        self.loss_f_s_log = []  # 新增
        self.loss_f_b_log = []  # 新增
        self.loss_BC_Q_log = []  # 已改
        self.loss_BC_Z_log = []  # 新增
        # self.loss_IC_h_log = [] # 删除
        self.loss_obs_Q_log = []  # 已改
        self.loss_obs_Z_log = []  # 已改
        self.loss_obs_S_log = []  # 新增
        self.loss_obs_A0_log = []  # 新增
        self.l2_Q_error_log = []  # 已改
        self.l2_Z_error_log = []  # 已改
        self.l2_U_error_log = []  # 新增
        self.l2_S_error_log = []  # 新增
        self.l2_A0_error_log = []  # 新增

        # Generate dicts for gradients storage
        self.dict_gradients_res_layers = self.generate_grad_dict(self.layers)
        self.dict_gradients_bcs_Q_layers = self.generate_grad_dict(self.layers)  # 已改
        self.dict_gradients_bcs_Z_layers = self.generate_grad_dict(self.layers)  # 新增
        # self.dict_gradients_ics_h_layers = self.generate_grad_dict(self.layers) # 删除
        if self.useObs:  # 新增
            self.dict_gradients_obs_Q_layers = self.generate_grad_dict(self.layers)
            self.dict_gradients_obs_Z_layers = self.generate_grad_dict(self.layers)
            self.dict_gradients_obs_S_layers = self.generate_grad_dict(self.layers)
            self.dict_gradients_obs_A0_layers = self.generate_grad_dict(self.layers)

        # Gradients Storage
        self.grad_res = []
        self.grad_bcs_Q = []  # 已改
        self.grad_bcs_Z = []  # 新增
        # self.grad_ics_h = [] # 删除
        self.grad_obs_Q = []  # 已改
        self.grad_obs_Z = []  # 新增
        self.grad_obs_S = []
        self.grad_obs_A0 = []
        for i in range(len(self.layers) - 1):
            self.grad_res.append(tf.gradients(self.loss_f, self.weights[i])[0])
            self.grad_bcs_Q.append(tf.gradients(self.loss_BC_Q, self.weights[i])[0])  # 已改
            self.grad_bcs_Z.append(tf.gradients(self.loss_BC_Z, self.weights[i])[0])  # 新增
            # self.grad_ics_h.append(tf.gradients(self.loss_IC_h, self.weights[i])[0]) # 删除
            if self.useObs:  # 新增
                self.grad_obs_Q.append(tf.gradients(self.loss_obs_Q, self.weights[i])[0])
                self.grad_obs_Z.append(tf.gradients(self.loss_obs_Z, self.weights[i])[0])
                self.grad_obs_S.append(tf.gradients(self.loss_obs_S, self.weights[i])[0])
                self.grad_obs_A0.append(tf.gradients(self.loss_obs_A0, self.weights[i])[0])

        self.adpative_constant_bcs_Q_list = []  # 已改
        self.adpative_constant_bcs_Q_log = []  # 已改
        self.adpative_constant_bcs_Z_list = []  # 新增
        self.adpative_constant_bcs_Z_log = []  # 新增
        self.adpative_constant_obs_Q_list = []  # obs新增
        self.adpative_constant_obs_Q_log = []
        self.adpative_constant_obs_Z_list = []
        self.adpative_constant_obs_Z_log = []
        self.adpative_constant_obs_S_list = []
        self.adpative_constant_obs_S_log = []
        self.adpative_constant_obs_A0_list = []
        self.adpative_constant_obs_A0_log = []

        for i in range(len(self.layers) - 1):
            self.adpative_constant_bcs_Q_list.append(
                tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_bcs_Q[i])))  # 已改
            self.adpative_constant_bcs_Z_list.append(
                tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_bcs_Z[i])))  # 新增
            if self.useObs:  # 已改
                self.adpative_constant_obs_Q_list.append(
                    tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_obs_Q[i])))
                self.adpative_constant_obs_Z_list.append(
                    tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_obs_Z[i])))
                self.adpative_constant_obs_S_list.append(
                    tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_obs_S[i])))
                self.adpative_constant_obs_A0_list.append(
                    tf.reduce_max(tf.abs(self.grad_res[i])) / tf.reduce_mean(tf.abs(self.grad_obs_A0[i])))

        self.adaptive_constant_bcs_Q = tf.reduce_max(tf.stack(self.adpative_constant_bcs_Q_list))  # 已改
        self.adaptive_constant_bcs_Z = tf.reduce_max(tf.stack(self.adpative_constant_bcs_Z_list))  # 新增
        if self.useObs:  # 新增
            self.adaptive_constant_obs_Q = tf.reduce_max(tf.stack(self.adpative_constant_obs_Q_list))
            self.adaptive_constant_obs_Z = tf.reduce_max(tf.stack(self.adpative_constant_obs_Z_list))
            self.adaptive_constant_obs_S = tf.reduce_max(tf.stack(self.adpative_constant_obs_S_list))
            self.adaptive_constant_obs_A0 = tf.reduce_max(tf.stack(self.adpative_constant_obs_A0_list))

        init = tf.global_variables_initializer()

        self.sess.run(init)

    # 学习率预热函数
    def get_lr_with_warmup(self, global_step, initial_lr, warmup_steps, total_steps):
        # warmup阶段：逐步增加学习率
        warmup_lr = initial_lr * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)

        # 进入衰减阶段：在 warmup 完成后，使用指数衰减
        decay_lr = tf.train.exponential_decay(
            initial_lr,  # 初始学习率
            global_step - warmup_steps,  # 当前全局步数（减去 warmup 步数）
            total_steps - warmup_steps,  # 衰减所需的总步数
            0.93,  # 衰减率
            staircase=False  # 是否以阶梯方式衰减
        )

        # 如果 global_step 小于 warmup_steps，则使用 warmup_lr，否则使用 decay_lr
        return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)


    # 创建每一层梯度字典
    def generate_grad_dict(self, layers):
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict



    def initialize_NN(self, layers, scope="default"):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            with tf.variable_scope(scope + "_layer{}".format(l)):
                W = self.xavier_init([layers[l], layers[l + 1]], scope=scope + "_W{}".format(l))
                b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=self.DTYPE),
                                name="b_{}".format(l), dtype=self.DTYPE)
                weights.append(W)
                biases.append(b)
        return weights, biases

    def xavier_init(self, size, scope="default"):
        in_dim = size[0]
        out_dim = size[1]

        with tf.name_scope(scope + "_xavier"):
            in_dim_tf = tf.constant(float(in_dim), dtype=self.DTYPE)
            out_dim_tf = tf.constant(float(out_dim), dtype=self.DTYPE)

            stddev = tf.sqrt(2.0 / (in_dim_tf + out_dim_tf))

            # 添加 tf.Print 以调试 stddev
            stddev = tf.Print(stddev, [stddev], message="Xavier stddev for {}: ".format(scope))

            # 用 stddev 初始化变量
            init = tf.random_normal([in_dim, out_dim], stddev=stddev, dtype=self.DTYPE)

            # 添加名称用于区分变量作用域
            return tf.Variable(init, dtype=self.DTYPE, name="W_" + scope)

    def compute_loss_landscape(self, grid_size=51, scale_range=0.3, feed_dict=None):
        if feed_dict is None:
            # 从train中复用tf_dict作为默认 (调整为不包含dummy等无关)
            feed_dict = {
                self.x_A_tf: self.x_A, self.x_B_tf: self.x_B, self.x_w_tf: self.x_w, self.x_a_tf: self.x_a,
                self.x_delta_S_tf: self.x_delta_S,
                self.x_u_BC_tf: self.x_u_BC, self.t_u_BC_tf: self.t_u_BC, self.Q_BC_tf: self.Q_BC,
                self.x_Z_BC_tf: self.x_Z_BC, self.t_Z_BC_tf: self.t_Z_BC, self.Z_BC_tf: self.Z_BC,
                self.x_BC_A_tf: self.x_BC_A, self.x_BC_B_tf: self.x_BC_B, self.x_BC_w_tf: self.x_BC_w,
                self.x_BC_a_tf: self.x_BC_a, self.x_BC_delta_S_tf: self.x_BC_delta_S,
                self.x_Z_BC_A_tf: self.x_Z_BC_A, self.x_Z_BC_B_tf: self.x_Z_BC_B,
                self.x_Z_BC_w_tf: self.x_Z_BC_w, self.x_Z_BC_a_tf: self.x_Z_BC_a,
                self.x_Z_BC_delta_S_tf: self.x_Z_BC_delta_S,
                self.x_u_obs_tf: self.x_u_obs, self.t_u_obs_tf: self.t_u_obs, self.Q_obs_tf: self.Q_obs,
                self.Z_obs_tf: self.Z_obs, self.S_obs_tf: self.S_obs, self.A0_obs_tf: self.A0_obs,
                self.x_A_obs_tf: self.x_A_obs, self.x_B_obs_tf: self.x_B_obs,
                self.x_delta_S_obs_tf: self.x_delta_S_obs, self.x_w_obs_tf: self.x_w_obs,
                self.x_a_obs_tf: self.x_a_obs,
                self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                self.dummy_tf: np.ones((self.t_f.shape[0], self.layers[-1])),
            }

        # 步骤1: 获取所有可训练变量
        trainable_vars = tf.trainable_variables()
        orig_params = self.sess.run(trainable_vars)  # list of numpy arrays

        # 步骤2: 生成两个随机方向 (与参数形状匹配)
        def generate_random_direction(params):
            direction = []
            for p in params:
                d = np.random.normal(size=p.shape).astype(np.float32)
                norm_p = np.linalg.norm(p.flatten())  # L2 norm of param
                norm_d = np.linalg.norm(d.flatten())
                d = d * (norm_p / norm_d) if norm_d != 0 else d  # 归一化到param norm
                direction.append(d)
            return direction

        dir1 = generate_random_direction(orig_params)
        dir2 = generate_random_direction(orig_params)

        # 步骤3: 创建网格
        alphas = np.linspace(-scale_range, scale_range, grid_size)
        betas = np.linspace(-scale_range, scale_range, grid_size)
        losses = np.zeros((grid_size, grid_size))

        # 步骤4: 创建 placeholders 和 assign_ops（仅一次，在循环外）
        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in trainable_vars]
        assign_ops = [v.assign(ph) for v, ph in zip(trainable_vars, placeholders)]

        # 步骤5: 对于每个网格点，扰动参数，计算损失
        start_time = time.time()
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # 计算新参数
                new_params = [orig + alpha * d1 + beta * d2 for orig, d1, d2 in zip(orig_params, dir1, dir2)]

                # 赋值新参数（使用 dict comprehension 创建 feed_assign）
                feed_assign = {ph: new_p for ph, new_p in zip(placeholders, new_params)}
                self.sess.run(assign_ops, feed_dict=feed_assign)

                # 计算损失
                loss_value = self.sess.run(self.loss, feed_dict=feed_dict)
                if np.isnan(loss_value) or np.isinf(loss_value):
                    losses[i, j] = 1e10  # 或 np.nan
                else:
                    losses[i, j] = min(loss_value, 1e6)  # clip 上限
                losses[i, j] = loss_value

            print(f"Row {i + 1}/{grid_size} completed. Time elapsed: {time.time() - start_time:.2f}s")

        # 步骤6: 恢复原参数
        feed_orig = {ph: orig_p for ph, orig_p in zip(placeholders, orig_params)}
        self.sess.run(assign_ops, feed_dict=feed_orig)

        # 保存数据到文件
        np.savez('结果/实验Z/有输入/缩放50/缩放adapt/最佳轮次/loss_landscape_data.npz', alphas=alphas, betas=betas, losses=losses)
        print("Loss landscape data saved to 'loss_landscape_data.npz'")

        return alphas, betas, losses

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def fwd_gradients(self, U, x):
        g = tf.gradients(U, x, grad_ys=self.dummy_tf)[0]
        return tf.gradients(g, self.dummy_tf)[0]


    def net_uh(self, x, t, A, B, w, a, delta_S):
        X = 2.0 * (tf.concat([x, t, A, B, w, a, delta_S], 1) - self.lb) / (
                self.ub - self.lb) - 1.0

        output5 = self.neural_net(X, self.weights, self.biases)
        Q = output5[:, 0:1]
        Z = output5[:, 1:2]
        S = output5[:, 2:3]
        A0 = output5[:, 3:4]

        return Q, Z, S, A0

    # 已修改
    def net_f(self, x_f, t_f, x_A_tf, x_B_tf, x_w_tf, x_a_tf, x_delta_S_tf):
        X_f = 2.0 * (tf.concat([x_f, t_f, x_A_tf, x_B_tf, x_w_tf, x_a_tf,
                                x_delta_S_tf], 1) - self.lb) / (self.ub - self.lb) - 1.0

        output5 = self.neural_net(X_f, self.weights, self.biases)
        Q = output5[:, 0:1]
        Z = output5[:, 1:2]
        S = output5[:, 2:3]
        A0 = output5[:, 3:4]

        Q_t = tf.gradients(Q, t_f)[0]
        Q_x = tf.gradients(Q, x_f)[0]

        Z_x = tf.gradients(Z, x_f)[0]
        Z_t = tf.gradients(Z, t_f)[0]

        S_t = tf.gradients(S, t_f)[0]
        S_x = tf.gradients(S, x_f)[0]
        A0_t = tf.gradients(A0, t_f)[0]

        eq1 = self.fun_r_mass( Q_x, Z_t)
        eq2 = self.fun_r_momentum(Q, Q_t, Q_x, Z_x, Z)
        eq3 = self.fun_sed(Q, S, Z_t, S_t, S_x, Q_x)
        eq4 = self.fun_bed(S, A0_t)

        return eq1, eq2, eq3, eq4

    def fun_r_mass(self, Q_x, Z_t):
        Q0 = 50
        return Q_x * Q0 + self.B * Z_t

    def fun_r_momentum(self, Q, Q_t, Q_x, Z_x, Z):
        Q0 = 50
        term1 = Q_t * Q0
        term2 = (self.g * self.A - (self.B * (Q ** 2 * Q0 * Q0) / (self.A ** 2))) * Z_x
        term3 = 2 * Q * Q0 / self.A * Q_x * Q0
        term4 = (Q ** 2 * Q0 * Q0) / (self.A ** 2) * self.A_x
        Z_safe = tf.maximum(Z, 1e-3)
        term5 = -self.g * Q * Q0 * Q * Q0 * self.n_star * self.n_star / (self.A * Z_safe ** (4 / 3))
        # 限制水深的最大最小值，防止数值不稳定。
        # h = tf.clip_by_value(h, clip_value_min=1e-4, clip_value_max=50)
        pde2_loss = term1 + term2 + term3 - term4 - term5

        return pde2_loss


    # 泥沙方程(不确定对不对，把A当作变量了，偏导用的链式法则,且要注意i：i+1，可能不对）
    def fun_sed(self, Q, S, Z_t, S_t, S_x, Q_x):
        loss = []
        pde3_loss = [0]
        for i in range(7):
            s = S * self.delta_S[:, i:i + 1] / 10  # 第i组悬沙含沙量
            s_1 = self.S_1 * self.delta_S[:, i:i + 1]  # 第i组悬沙含沙量
            w = self.w * self.delta_S[:, i:i + 1]
            s_t = tf.gradients(s, self.t_f_tf)[0]
            s_x = tf.gradients(s, self.x_f_tf)[0]

            term1 = self.B * s * Z_t
            term2 = self.A * s_t
            term3 = Q * s_x
            term4 = s * Q_x
            term5 = - self.B * w * self.a[:, i:i + 1] * (s - s_1)
            if i == 0:
                loss1 = term1 + term2 + term3 + term4 - term5
            if i == 1:
                loss2 = term1 + term2 + term3 + term4 - term5
            if i == 2:
                loss3 = term1 + term2 + term3 + term4 - term5
            if i == 3:
                loss4 = term1 + term2 + term3 + term4 - term5
            if i == 4:
                loss5 = term1 + term2 + term3 + term4 - term5
            if i == 5:
                loss6 = term1 + term2 + term3 + term4 - term5
            if i == 6:
                loss7 = term1 + term2 + term3 + term4 - term5

        pde3_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
        return pde3_loss

    # 河床方程损失
    def fun_bed(self, S, A0_t):
        loss = []
        loss_sum = [0]
        for i in range(7):
            s = S * self.delta_S[:, i:i + 1] / 10  # 第i组悬沙含沙量
            w = self.w * self.delta_S[:, i:i + 1]
            term = self.B * w * self.a[:, i:i + 1] * (s - self.S_1)
            if i == 0:
                loss1 = term
            if i == 1:
                loss2 = term
            if i == 2:
                loss3 = term
            if i == 3:
                loss4 = term
            if i == 4:
                loss5 = term
            if i == 5:
                loss6 = term
            if i == 6:
                loss7 = term
        pde4_loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7) - self.P * A0_t
        return pde4_loss


    def train(self, num_epochs):

        tf_dict = {  # self.x_h_IC_tf: self.x_h_IC, self.t_h_IC_tf: self.t_h_IC, self.h_IC_tf: self.h_IC,
            self.x_A_tf: self.x_A, self.x_B_tf: self.x_B, self.x_w_tf: self.x_w, self.x_a_tf: self.x_a,
            self.x_delta_S_tf: self.x_delta_S,
            self.x_u_BC_tf: self.x_u_BC, self.t_u_BC_tf: self.t_u_BC, self.Q_BC_tf: self.Q_BC,  # 已改
            self.x_h_BC_tf: self.x_h_BC, self.t_h_BC_tf: self.t_h_BC,  # 已改
            self.x_Z_BC_tf: self.x_Z_BC, self.t_Z_BC_tf: self.t_Z_BC, self.Z_BC_tf: self.Z_BC,  # 新增
            self.x_BC_A_tf: self.x_BC_A, self.x_BC_B_tf: self.x_BC_B, self.x_BC_w_tf: self.x_BC_w,
            self.x_BC_a_tf: self.x_BC_a, self.x_BC_delta_S_tf: self.x_BC_delta_S,
            self.x_Z_BC_A_tf: self.x_Z_BC_A, self.x_Z_BC_B_tf: self.x_Z_BC_B,
            self.x_Z_BC_w_tf: self.x_Z_BC_w, self.x_Z_BC_a_tf: self.x_Z_BC_a,
            self.x_Z_BC_delta_S_tf: self.x_Z_BC_delta_S,
            self.x_u_obs_tf: self.x_u_obs, self.t_u_obs_tf: self.t_u_obs, self.Q_obs_tf: self.Q_obs,  # 已改
            self.x_h_obs_tf: self.x_h_obs, self.t_h_obs_tf: self.t_h_obs, self.Z_obs_tf: self.Z_obs,  # 已改
            self.S_obs_tf: self.S_obs, self.A0_obs_tf: self.A0_obs,  # 新增
            self.x_A_obs_tf: self.x_A_obs, self.x_B_obs_tf: self.x_B_obs,
            self.x_delta_S_obs_tf: self.x_delta_S_obs, self.x_w_obs_tf: self.x_w_obs,
            self.x_a_obs_tf: self.x_a_obs,
            self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
            self.dummy_tf: np.ones((self.t_f.shape[0], self.layers[-1])),

            self.adaptive_constant_bcs_Q_tf: self.adaptive_constant_bcs_Q_val,  # 已改
            self.adaptive_constant_bcs_Z_tf: self.adaptive_constant_bcs_Z_val,  # 新增
            self.adaptive_constant_obs_Q_tf: self.adaptive_constant_obs_Q_val,  # obs新增
            self.adaptive_constant_obs_Z_tf: self.adaptive_constant_obs_Z_val,
            self.adaptive_constant_obs_S_tf: self.adaptive_constant_obs_S_val,
            self.adaptive_constant_obs_A0_tf: self.adaptive_constant_obs_A0_val,
        }
        train_loss = []
        train_Q_error = []
        min_loss = 1

        excel_path = "结果/实验Z/有输入/缩放50/缩放adapt/最佳轮次/log.xlsx"

        # 如果文件不存在，创建表头
        if not os.path.exists(excel_path):
            df = pd.DataFrame(columns=[
                'Epoch', 'Loss', 'Learning_Rate', 'Loss_BC_Q', 'Loss_BC_Z',
                'Loss_obs_Q', 'Loss_obs_Z', 'Loss_obs_S', 'Loss_f_c',
                'Loss_f_m', 'Loss_f_s', 'Loss_f_b',
                'Error_Q', 'Error_Z', 'Error_S', 'Error_A0'
            ])
            df.to_excel(excel_path, index=False)

        for it in range(num_epochs):

            start_time = time.time()
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)

                if loss_value < min_loss:
                    min_loss = loss_value
                    num1 = it
                train_loss.append(loss_value.item())  # 记录损失

                learning_rate = self.sess.run(self.learning_rate)
                print('It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                      % (it, loss_value, elapsed, learning_rate))

                # 网络预测已改，误差已改
                Q_pred, Z_pred, S_pred, A0_pred = self.predict(self.X_star[:, 0:1], self.X_star[:, 1:2])
                error_Q = np.linalg.norm(self.Q_star - Q_pred, 2) / np.linalg.norm(self.Q_star, 2)
                error_Z = np.linalg.norm(self.Z_star - Z_pred, 2) / np.linalg.norm(self.Z_star, 2)
                error_S = np.linalg.norm(self.S_star - S_pred, 2) / np.linalg.norm(self.S_star, 2)
                error_A0 = np.linalg.norm(self.A0_star - A0_pred, 2) / np.linalg.norm(self.A0_star, 2)

                train_Q_error.append(error_Q.item())  # 记录误差

                with open("结果/实验Z/有输入/缩放50/缩放adapt/最佳轮次/train_loss.txt", 'w') as train_los:
                    train_los.write(str(train_loss))

                with open("结果/实验Z/有输入/缩放50/缩放adapt/最佳轮次/train_Q_error.txt", 'w') as train_ac:
                    train_ac.write(str(train_Q_error))

                if self.useObs:  # 新增两PDE损失项,修改BC损失中Q和S,增加BC损失中Z，删去IC  没有A0
                    loss_BC_Q, loss_BC_Z, loss_obs_Q, loss_obs_Z, loss_obs_S, loss_f_c, loss_f_m, loss_f_s, loss_f_b = \
                        self.sess.run([self.loss_BC_Q, self.loss_BC_Z, self.loss_obs_Q, self.loss_obs_Z,
                                       self.loss_obs_S,
                                       self.loss_f_c, self.loss_f_m, self.loss_f_s, self.loss_f_b], tf_dict)

                else:
                    loss_BC_Q, loss_BC_Z, loss_f_c, loss_f_m, loss_f_s, loss_f_b = \
                        self.sess.run([self.loss_BC_Q, self.loss_BC_Z, self.loss_f_c, self.loss_f_m,
                                       self.loss_f_s, self.loss_f_b],
                                      tf_dict)
                    print(
                        'Loss_BC_Q: %.3e, Loss_BC_Z: %.3e, Loss_f_c: %.3e, Loss_f_m: %.3e,Loss_f_s: '
                        '%.3e,Loss_f_b: %.3e, Error Q: %.3e, Error Z: %.3e, '
                        'Error S: %.3e, Error A0: %.3e'
                        % (loss_BC_Q, loss_BC_Z, loss_f_c, loss_f_s, loss_f_b, loss_f_m, error_Q, error_Z,
                           error_S, error_A0))

                self.loss_f_c_log.append(loss_f_c)
                self.loss_f_m_log.append(loss_f_m)
                self.loss_f_c_log.append(loss_f_s)  # 新增
                self.loss_f_m_log.append(loss_f_b)  # 新增
                self.loss_BC_Q_log.append(loss_BC_Q)  # 已改
                self.loss_BC_Z_log.append(loss_BC_Z)  # 新增
                # self.loss_IC_h_log.append(loss_IC_h)
                if self.useObs:  # obs新增
                    self.loss_obs_Q_log.append(loss_obs_Q)
                    self.loss_obs_Z_log.append(loss_obs_Z)
                    self.loss_obs_S_log.append(loss_obs_S)
                    # self.loss_obs_A0_log.append(loss_obs_A0)
                self.l2_Q_error_log.append(error_Q)  # 已改
                self.l2_Z_error_log.append(error_Z)  # 已改
                self.l2_S_error_log.append(error_S)  # 新增
                self.l2_A0_error_log.append(error_A0)  # 新增

                # Compute the adaptive constant
                adaptive_constant_bcs_Q_val = self.sess.run(self.adaptive_constant_bcs_Q, tf_dict)
                self.adaptive_constant_bcs_Q_val = adaptive_constant_bcs_Q_val * \
                                                   (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_Q_val
                self.adpative_constant_bcs_Q_log.append(self.adaptive_constant_bcs_Q_val)
                # 新增BC_Z
                adaptive_constant_bcs_Z_val = self.sess.run(self.adaptive_constant_bcs_Z, tf_dict)
                self.adaptive_constant_bcs_Z_val = adaptive_constant_bcs_Z_val * \
                                                   (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_Z_val
                self.adpative_constant_bcs_Z_log.append(self.adaptive_constant_bcs_Z_val)
                if self.useObs:
                    adaptive_constant_obs_Q_val = self.sess.run(self.adaptive_constant_obs_Q, tf_dict)
                    self.adaptive_constant_obs_Q_val = adaptive_constant_obs_Q_val * \
                                                       (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_Q_val
                    self.adpative_constant_obs_Q_log.append(self.adaptive_constant_obs_Q_val)

                    adaptive_constant_obs_Z_val = self.sess.run(self.adaptive_constant_obs_Z, tf_dict)
                    self.adaptive_constant_obs_Z_val = adaptive_constant_obs_Z_val * \
                                                       (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_Z_val
                    self.adpative_constant_obs_Z_log.append(self.adaptive_constant_obs_Z_val)


                    adaptive_constant_obs_S_val = self.sess.run(self.adaptive_constant_obs_S, tf_dict)
                    self.adaptive_constant_obs_S_val = adaptive_constant_obs_S_val * \
                                                       (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_S_val
                    self.adpative_constant_obs_S_log.append(self.adaptive_constant_obs_S_val)

                    adaptive_constant_obs_A0_val = self.sess.run(self.adaptive_constant_obs_A0, tf_dict)
                    self.adaptive_constant_obs_A0_val = adaptive_constant_obs_A0_val * \
                                                        (
                                                                    1.0 - self.beta) + self.beta * self.adaptive_constant_obs_A0_val
                    self.adpative_constant_obs_A0_log.append(self.adaptive_constant_obs_A0_val)

                if self.useObs:  # BC已改已增加 IC已删
                    print(
                        "constant_bcs_Q_val: {:.3f}, constant_bcs_Z_val: {:.3f}, "
                        "constant_obs_Q_val: {:.3f}, constant_obs_Z_val: {:.3f}, constant_obs_U_val: {:.3f}, "
                        "constant_obs_S_val: {:.3f}".format(
                            self.adaptive_constant_bcs_Q_val,
                            self.adaptive_constant_bcs_Z_val, self.adaptive_constant_obs_Q_val,
                            self.adaptive_constant_obs_Z_val, self.adaptive_constant_obs_U_val,
                            self.adaptive_constant_obs_S_val))

                    # 每50轮记录一下  没有A0
                    if it % 50 == 0:
                        # 准备数据
                        new_data = {
                            'Epoch': it,
                            'Loss': loss_value,
                            'Learning_Rate': learning_rate,
                            'Loss_BC_Q': loss_BC_Q,
                            'Loss_BC_Z': loss_BC_Z,
                            'Loss_obs_Q': loss_obs_Q,
                            'Loss_obs_Z': loss_obs_Z,
                            'Loss_obs_S': loss_obs_S,
                            'Loss_f_c': loss_f_c,
                            'Loss_f_m': loss_f_m,
                            'Loss_f_s': loss_f_s,
                            'Loss_f_b': loss_f_b,
                            'Error_Q': error_Q,
                            'Error_Z': error_Z,
                            'Error_S': error_S,
                            'Error_A0': error_A0
                        }

                        # 读取现有数据并追加
                        df_existing = pd.read_excel(excel_path)
                        df_new = pd.DataFrame([new_data])
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        df_combined.to_excel(excel_path, index=False)

                    # 每10轮打印
                    print(
                        'Loss_BC_Q: %.3e, Loss_BC_Z: %.3e, Loss_obs_Q: %.3e, Loss_obs_Z: %.3e,  Loss_obs_S: %.3e, '
                        'Loss_f_c: %.3e, Loss_f_m: %.3e,Loss_f_s: %.3e,Loss_f_b: %.3e, Error Q: %.3e, Error Z: %.3e, '
                        ' Error S: %.3e, Error A0: %.3e '
                        % (
                            loss_BC_Q,  loss_BC_Z, loss_obs_Q, loss_obs_Z, loss_obs_S, loss_f_c, loss_f_m, loss_f_s,
                            loss_f_b, error_Q, error_Z,  error_S, error_A0))

                else:
                    print("constant_bcs_Q_val: {:.3f}, constant_bcs_S_val: {:.3f}, constant_bcs_Z_val: {:.3f}".format(
                        self.adaptive_constant_bcs_Q_val, self.adaptive_constant_bcs_S_val,
                        self.adaptive_constant_bcs_Z_val))


        start_time = time.time()
        self.compute_loss_landscape(grid_size=51, scale_range=0.3, feed_dict=tf_dict)  # 使用train中的tf_dict

    def predict(self, x_star, t_star):  # 网络预测输出5个值，已改

        tf_dict = {self.x_u_tf: x_star, self.t_u_tf: t_star,
                   self.x_A_tf: self.x_A, self.x_B_tf: self.x_B,
                   self.x_w_tf: self.x_w, self.x_a_tf: self.x_a,
                   self.x_delta_S_tf: self.x_delta_S,
                   self.x_h_tf: x_star, self.t_h_tf: t_star}

        Q_star = self.sess.run(self.Q_pred, tf_dict)
        Z_star = self.sess.run(self.Z_pred, tf_dict)
        S_star = self.sess.run(self.S_pred, tf_dict)
        A0_star = self.sess.run(self.A0_pred, tf_dict)

        return Q_star, Z_star, S_star, A0_star

    def save_share_NN(self, filepath_wb, filepath_op):

        weights = self.sess.run(self.weights_shared)
        biases = self.sess.run(self.biases_shared)

        # # 使用 tf.train.Checkpoint 保存模型和优化器
        # checkpoint = tf.train.Checkpoint(
        #     optimizer=self.optimizer_Adam,
        #     weights=self.weights,
        #     biases=self.biases
        # )
        # checkpoint.save(filepath_op)
        # print(f"Checkpoint saved at {filepath_op}.")

        with open(filepath_wb, 'wb') as f:
            pickle.dump([weights, biases], f)
            print("Save u h NN parameters successfully...")

    def save_Q_NN(self, filepath_wb, filepath_op):

        weights = self.sess.run(self.weights_head1)
        biases = self.sess.run(self.biases_head1)

        # # 使用 tf.train.Checkpoint 保存模型和优化器
        # checkpoint = tf.train.Checkpoint(
        #     optimizer=self.optimizer_Adam,
        #     weights=self.weights,
        #     biases=self.biases
        # )
        # checkpoint.save(filepath_op)
        # print(f"Checkpoint saved at {filepath_op}.")

        with open(filepath_wb, 'wb') as f:
            pickle.dump([weights, biases], f)
            print("Save u h NN parameters successfully...")

    def save_Z_NN(self, filepath_wb, filepath_op):

        weights = self.sess.run(self.weights_head2)
        biases = self.sess.run(self.biases_head2)

        # # 使用 tf.train.Checkpoint 保存模型和优化器
        # checkpoint = tf.train.Checkpoint(
        #     optimizer=self.optimizer_Adam,
        #     weights=self.weights,
        #     biases=self.biases
        # )
        # checkpoint.save(filepath_op)
        # print(f"Checkpoint saved at {filepath_op}.")

        with open(filepath_wb, 'wb') as f:
            pickle.dump([weights, biases], f)
            print("Save u h NN parameters successfully...")

    def save_S_NN(self, filepath_wb, filepath_op):

        weights = self.sess.run(self.weights_head3)
        biases = self.sess.run(self.biases_head3)

        # # 使用 tf.train.Checkpoint 保存模型和优化器
        # checkpoint = tf.train.Checkpoint(
        #     optimizer=self.optimizer_Adam,
        #     weights=self.weights,
        #     biases=self.biases
        # )
        # checkpoint.save(filepath_op)
        # print(f"Checkpoint saved at {filepath_op}.")

        with open(filepath_wb, 'wb') as f:
            pickle.dump([weights, biases], f)
            print("Save u h NN parameters successfully...")

    def load_NN(self, filepath_wb, filepath_op, layers):
        weights = []
        biases = []
        num_layers = len(layers)

        # checkpoint = tf.train.Checkpoint(
        #     optimizer=self.optimizer_Adam,
        #     weights=self.weights,
        #     biases=self.biases
        # )
        # checkpoint.restore(tf.train.latest_checkpoint(filepath_op))
        # print(f"Loaded TensorFlow checkpoint from {filepath_op}.")

        with open(filepath_wb, 'rb') as f:
            uh_weights, uh_biases = pickle.load(f)

            assert num_layers == (len(uh_weights) + 1)

            for num in range(0, num_layers - 1):
                weights.append(tf.Variable(uh_weights[num]))
                biases.append(tf.Variable(uh_biases[num]))
            print(" - Load NN parameters successfully...")
        return weights, biases

    def save_weight(self, filepath):
        if self.useObs:
            weight_array = np.vstack([np.array(self.adpative_constant_bcs_Q_log),  # 已改
                                      np.array(self.adpative_constant_bcs_S_log),  # 已改
                                      np.array(self.adpative_constant_bcs_Z_log),  # 新增
                                      # np.array(self.adpative_constant_ics_h_log),
                                      np.array(self.adpative_constant_obs_Q_log),  # 已改
                                      np.array(self.adpative_constant_obs_Z_log),
                                      np.array(self.adpative_constant_obs_U_log),
                                      np.array(self.adpative_constant_obs_S_log)
                                      # np.array(self.adpative_constant_obs_A0_log)
                                     ])
        else:
            weight_array = np.vstack([np.array(self.adpative_constant_bcs_Q_log),  # 已改
                                      np.array(self.adpative_constant_bcs_S_log),  # 已改
                                      np.array(self.adpative_constant_bcs_S_log)  # 新增
                                      # np.array(self.adpative_constant_ics_h_log)
                                      ])
        np.savetxt(filepath, weight_array.T, fmt='%1.4e')

    def load_weight(self, filepath):
        weight_array = np.loadtxt(filepath)
        weight_array = weight_array.T

        if self.useObs:
            return weight_array[0, -1], weight_array[1, -1], weight_array[2, -1], weight_array[3, -1], weight_array[
                4, -1], weight_array[5, -1], weight_array[6, -1]
        else:
            return weight_array[0, -1], weight_array[1, -1], weight_array[2, -1]


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
