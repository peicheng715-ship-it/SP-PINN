import numpy as np
from datetime import datetime
import time
import tensorflow as tf
print(tf.__version__)
import pickle
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1234)
tf.set_random_seed(1234)

class SVE:
    DTYPE = tf.float32

    # Initialize the class
    def __init__(self,
                 X_u_BC, X_h_BC, x_BC_A, x_BC_B, x_BC_w, x_BC_a, x_BC_delta_S,  # 边界入口输入（x，t，参数）
                 X_Z_BC, x_Z_BC_A, x_Z_BC_B, x_Z_BC_w, x_Z_BC_a, x_Z_BC_delta_S,  # 边界出口输入（x，t，参数）
                 X_u_obs, X_h_obs, x_A_obs, x_B_obs, x_delta_S_obs, x_w_obs, x_a_obs,  # 观测输入（x，t，参数）
                 X_f, x_A, x_B, x_w, x_a, x_delta_S,  # 方程输入（x，t，参数）
                 Q_BC, Z_BC,  # 边界 真实值
                 Q_obs, Z_obs, S_obs, A0_obs,  # 观测真实值
                 layers,
                 lb, ub,
                 X_star, Q_star, Z_star, S_star, A0_star, n_star,   # 方程解的真实值
                 B, A, w, a, S_1, P, delta_S, n, A_x,  # 损失函数使用
                 lr=2e-4,
                 ExistModel=0, uhDir='', wDir='', useObs=True):

        self.B = B  # 水面宽度
        self.A = A  # 断面过水面积
        # w和a待定
        self.w = w
        self.a = a  # 恢复饱和系数
        self.S_1 = S_1  # 挟沙力
        self.g = tf.constant(9.81, dtype=self.DTYPE)
        self.delta_S = delta_S  # 悬沙级配
        self.P = tf.constant(P, dtype=self.DTYPE)  # 床沙干密度 1400
        self.n = tf.constant(n, dtype=self.DTYPE)
        self.A_x = A_x
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

        # 边界BC入口出口输入（x，t）
        self.x_u_BC = X_u_BC[:, 0:1]  # 实际使用
        self.t_u_BC = X_u_BC[:, 1:2]  # 实际使用
        self.x_h_BC = X_h_BC[:, 0:1]  # 没用
        self.t_h_BC = X_h_BC[:, 1:2]  # 没用
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

        # 边界和观测真值
        self.Q_BC = Q_BC  # 已改
        self.Z_BC = Z_BC  # 新增
        self.Q_obs = Q_obs
        self.Z_obs = Z_obs
        self.S_obs = S_obs
        self.A0_obs = A0_obs

        # layers
        self.layers = layers
        if ExistModel == 0:
            self.weights, self.biases = self.initialize_NN(layers)

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
        # 暂未使用
        self.x_h_tf = tf.placeholder(self.DTYPE, shape=[None, self.x_f.shape[1]])
        self.t_h_tf = tf.placeholder(self.DTYPE, shape=[None, self.t_f.shape[1]])

        # 边界BC入口输入占位符（x,t）    BC入口及出口真实值（Q,Z）
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
        self.eq1_pred, self.eq2_pred, self.eq3_pred, self.eq4_pred = self.net_f(self.x_f_tf, self.t_f_tf, self.x_A_tf,
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
        self.loss_BCs = 0.001 * self.loss_BC_Q + 100 * self.loss_BC_Z

        # 损失函数删掉了IC
        self.loss = 10 * self.loss_f + self.loss_BCs

        if self.useObs:  # 新增
            self.loss_obs_Q = tf.reduce_mean(tf.square(self.Q_obs_tf - self.Q_obs_pred))
            self.loss_obs_Z = tf.reduce_mean(tf.square(self.Z_obs_tf - self.Z_obs_pred))
            self.loss_obs_S = tf.reduce_mean(tf.square(self.S_obs_tf - self.S_obs_pred))
            self.loss_obs_A0 = tf.reduce_mean(tf.square(self.A0_obs_tf - self.A0_obs_pred))
            self.loss_obs = 0.001 *  self.loss_obs_Q \
                            + 100 * self.loss_obs_Z \
                            + 1000 *self.loss_obs_S \
                            + self.loss_obs_A0
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
        gradients, variables = zip(*self.optimizer_Adam.compute_gradients(self.loss))
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)  # 可调整 clip_norm
        self.train_op_Adam = self.optimizer_Adam.apply_gradients(
            zip(clipped_gradients, variables),
            global_step=self.global_step
        )

        init = tf.global_variables_initializer()
        self.sess.run(init)

    # 学习率预热函数
    def get_lr_with_warmup(self, global_step, initial_lr, warmup_steps, total_steps):
        # warmup阶段：逐步增加学习率
        warmup_lr = initial_lr * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
        decay_lr = tf.train.exponential_decay(
            initial_lr,  # 初始学习率
            global_step - warmup_steps,  # 当前全局步数（减去 warmup 步数）
            total_steps - warmup_steps,  # 衰减所需的总步数
            0.93,  # 衰减率
            staircase=False  # 是否以阶梯方式衰减
        )
        return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)

    # 创建每一层梯度字典
    def generate_grad_dict(self, layers):
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            # 更小的初始化范围
            W = tf.Variable(tf.random_normal([layers[l], layers[l + 1]], stddev=0.01), dtype=self.DTYPE)
            b = tf.Variable(tf.random_normal([1, layers[l + 1]], stddev=0.01), dtype=self.DTYPE)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]  # 权重矩阵的输入维度和输出维度分别存储
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))  # 计算标准差
        # 生成遵循截断正态分布的权重矩阵
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=self.DTYPE)

    def neural_net(self, X, weights, biases):
        H = X
        for l in range(len(weights) - 1):
            W = weights[l]
            b = biases[l]
            H = tf.matmul(H, W) + b
            H = tf.nn.leaky_relu(H, alpha=0.01)  # LeakyReLU 防止梯度消失
        # 最后一层不加激活（回归问题）
        Y = tf.matmul(H, weights[-1]) + biases[-1]
        return Y

    def net_uh(self, x, t, A, B, w, a, delta_S):
        X = 2.0 * (tf.concat([x, t, A, B, w, a, delta_S], 1) - self.lb) / (
                self.ub - self.lb) - 1.0

        output4 = self.neural_net(X, self.weights, self.biases)
        Q = output4[:, 0:1]
        Z = output4[:, 1:2]
        S = output4[:, 2:3]
        A0 = output4[:, 3:4]

        return Q, Z, S, A0

    # 已修改
    def net_f(self, x_f, t_f, x_A_tf, x_B_tf, x_w_tf, x_a_tf, x_delta_S_tf):
        X_f = 2.0 * (tf.concat([x_f, t_f, x_A_tf, x_B_tf, x_w_tf, x_a_tf,
                                x_delta_S_tf], 1) - self.lb) / (self.ub - self.lb) - 1.0
        output4 = self.neural_net(X_f, self.weights, self.biases)
        Q = output4[:, 0:1]
        Z = output4[:, 1:2]
        S = output4[:, 2:3]
        A0 = output4[:, 3:4]

        Q_t = tf.gradients(Q, t_f)[0]
        Q_x = tf.gradients(Q, x_f)[0]

        Z_x = tf.gradients(Z, x_f)[0]
        Z_t = tf.gradients(Z, t_f)[0]

        S_t = tf.gradients(S, t_f)[0]
        S_x = tf.gradients(S, x_f)[0]

        A0_t = tf.gradients(A0, t_f)[0]
        eq1 = self.fun_r_mass(Q_x, Z_t)
        eq2 = self.fun_r_momentum(Q, Q_t, Q_x, Z_x, Z)
        eq3 = self.fun_sed(Q, S, Z_t, S_t, S_x, Q_x)
        eq4 = self.fun_bed(S, A0_t)

        return eq1, eq2, eq3, eq4

    def fun_r_mass(self, Q_x, Z_t):
        Q0 = 500
        return Q_x * Q0 + self.B * Z_t

    def fun_r_momentum(self, Q, Q_t, Q_x, Z_x, Z):
        Q0 = 500
        term1 = Q_t * Q0
        term2 = (self.g * self.A - (self.B * (Q ** 2* Q0* Q0) / (self.A ** 2))) * Z_x
        term3 = 2 * Q* Q0 / self.A * Q_x* Q0
        term4 = (Q ** 2* Q0* Q0) / (self.A ** 2) * self.A_x
        Z_safe = tf.maximum(Z, 1e-3)
        term5 = -self.g * Q* Q0 * Q* Q0 * self.n_star * self.n_star / (self.A * Z_safe ** (4 / 3))
        # 限制水深的最大最小值，防止数值不稳定。
        # h = tf.clip_by_value(h, clip_value_min=1e-4, clip_value_max=50)
        pde2_loss = term1 + term2 + term3 - term4 - term5

        return pde2_loss

    def fun_sed(self, Q, S, Z_t, S_t, S_x, Q_x):
        loss = []
        pde3_loss = [0]
        for i in range(7):
            # s_x = S_x * self.delta_S[:, i:i + 1]  # 第i组悬沙含沙量对x的偏导
            # s_t = S_t * self.delta_S[:, i:i + 1]

            s = S * self.delta_S[:, i:i + 1] / 10  # 第i组悬沙含沙量
            s_1 = self.S_1 * self.delta_S[:, i:i + 1]  # 第i组悬沙含沙量
            w = self.w * self.delta_S[:, i:i + 1]
            s_t = tf.gradients(s, self.t_f_tf)[0]
            s_x = tf.gradients(s, self.x_f_tf)[0]
            # 求分组恢复饱和系数

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

    # def fun_sed(self, Q, S, Z_t, S_t, S_x, Q_x):
    #     loss = []
    #     pde3_loss = [0]
    #     s_t = tf.gradients(S, self.t_f_tf)[0]
    #     s_x = tf.gradients(S, self.x_f_tf)[0]
    #
    #
    #     term1 = self.B * S * Z_t
    #     term2 = self.A * s_t
    #     term3 = Q * s_x * 106.4
    #     term4 = S * Q_x * 106.4
    #     term5 = - self.B * self.w * 0.01 * (S - self.S_1)  # 0.01是恢复饱和系数
    #
    #     loss = term1 + term2 + term3 + term4 - term5
    #
    #     pde3_loss = loss
    #     return pde3_loss

    # 河床方程损失
    def fun_bed(self, S, A0_t):
        loss = []
        loss_sum = [0]
        for i in range(7):
            s = S * self.delta_S[:, i:i + 1] / 10 # 第i组悬沙含沙量
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

    # def fun_bed(self, S, A0_t):
    #     loss = []
    #     loss_sum = [0]
    #     term = self.B * self.w * 0.01 * (S - self.S_1)
    #     pde4_loss = term - self.P * A0_t
    #     return pde4_loss

    def callback_obs(self, loss, loss_f_c, loss_f_m, loss_f_s, loss_f_b, loss_BC_Q, loss_BC_Z,
                     loss_obs_Q, loss_obs_Z, loss_obs_S, loss_obs_A0):
        self.count = self.count + 1
        print('{} th iterations, Loss: {:.3e}, Loss_f_c: {:.3e}, Loss_f_m: {:.3e}'.format(self.count, loss, loss_f_c,
                                                                                          loss_f_m))
        self.loss_f_c_log.append(loss_f_c)
        self.loss_f_m_log.append(loss_f_m)
        self.loss_f_s_log.append(loss_f_s)  # 新增
        self.loss_f_b_log.append(loss_f_b)  # 新增
        self.loss_BC_Q_log.append(loss_BC_Q)  # 已改
        # self.loss_BC_S_log.append(loss_BC_S)  # 已改
        self.loss_BC_Z_log.append(loss_BC_Z)  # 新增
        # self.loss_IC_h_log.append(loss_IC_h) # 删除
        self.loss_obs_Q_log.append(loss_obs_Q)  # obs新增
        self.loss_obs_Z_log.append(loss_obs_Z)
        # self.loss_obs_U_log.append(loss_obs_U)
        self.loss_obs_S_log.append(loss_obs_S)
        self.loss_obs_A0_log.append(loss_obs_A0)

    def callback(self, loss, loss_f_c, loss_f_m, loss_f_s, loss_f_b, loss_BC_Q,  loss_BC_Z):
        self.count = self.count + 1
        print('{} th iterations, Loss: {:.3e}, Loss_f_c: {:.3e}, Loss_f_m: {:.3e}'.format(self.count, loss, loss_f_c,
                                                                                          loss_f_m))
        self.loss_f_c_log.append(loss_f_c)
        self.loss_f_m_log.append(loss_f_m)
        self.loss_f_s_log.append(loss_f_s)  # 新曾
        self.loss_f_b_log.append(loss_f_b)  # 新增
        self.loss_BC_Q_log.append(loss_BC_Q)  # 已改
        # self.loss_BC_S_log.append(loss_BC_S)  # 已改
        self.loss_BC_Z_log.append(loss_BC_Z)  # 删除
        # self.loss_IC_h_log.append(loss_IC_h)

    def compute_loss_landscape(self, grid_size=51, scale_range=0.3, feed_dict=None):
        """
        计算2D损失景观数据。
        :param grid_size: 网格分辨率 (e.g., 51 -> 51x51 点)
        :param scale_range: alpha/beta 范围 (-scale_range 到 scale_range)
        :param feed_dict: 用于计算损失的feed_dict (从train中复用)
        :return: alphas, betas, losses (3个numpy数组)
        """
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
        np.savez('结果/损失景观不缩放2/loss_landscape_data.npz', alphas=alphas, betas=betas, losses=losses)
        print("Loss landscape data saved to 'loss_landscape_data.npz'")

        return alphas, betas, losses

    def train(self, num_epochs):
        tf_dict = {  # self.x_h_IC_tf: self.x_h_IC, self.t_h_IC_tf: self.t_h_IC, self.h_IC_tf: self.h_IC,
            self.x_A_tf: self.x_A, self.x_B_tf: self.x_B, self.x_w_tf: self.x_w, self.x_a_tf: self.x_a,
            self.x_delta_S_tf: self.x_delta_S,
            self.x_u_BC_tf: self.x_u_BC, self.t_u_BC_tf: self.t_u_BC, self.Q_BC_tf: self.Q_BC,  # 已改
            self.x_h_BC_tf: self.x_h_BC, self.t_h_BC_tf: self.t_h_BC,   # 已改
            self.x_Z_BC_tf: self.x_Z_BC, self.t_Z_BC_tf: self.t_Z_BC, self.Z_BC_tf: self.Z_BC,  # 新增
            self.x_BC_A_tf: self.x_BC_A, self.x_BC_B_tf: self.x_BC_B,self.x_BC_w_tf: self.x_BC_w,
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
            # self.adaptive_constant_bcs_S_tf: self.adaptive_constant_bcs_S_val,  # 已改
            self.adaptive_constant_bcs_Z_tf: self.adaptive_constant_bcs_Z_val,  # 新增
            # self.adaptive_constant_ics_h_tf: self.adaptive_constant_ics_h_val,
            self.adaptive_constant_obs_Q_tf: self.adaptive_constant_obs_Q_val,  # obs新增
            self.adaptive_constant_obs_Z_tf: self.adaptive_constant_obs_Z_val,
            # self.adaptive_constant_obs_U_tf: self.adaptive_constant_obs_U_val,
            self.adaptive_constant_obs_S_tf: self.adaptive_constant_obs_S_val,
            self.adaptive_constant_obs_A0_tf: self.adaptive_constant_obs_A0_val,
        }
        train_loss = []
        train_Q_error = []
        min_loss = 1
        min_error = 1

        for it in range(num_epochs):

            start_time = time.time()
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
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
                # error_U = np.linalg.norm(self.U_star - U_pred, 2) / np.linalg.norm(self.U_star, 2)
                error_S = np.linalg.norm(self.S_star - S_pred, 2) / np.linalg.norm(self.S_star, 2)
                error_A0 = np.linalg.norm(self.A0_star - A0_pred, 2) / np.linalg.norm(self.A0_star, 2)

                np.savez(
                    '结果/FNN/best_predictions.npz',
                    Q_pred=Q_pred,
                    Z_pred=Z_pred,
                    S_pred=S_pred,
                    A0_pred=A0_pred,
                    epoch=it,
                    loss=loss_value
                )


                if error_Q < min_error:
                    min_error = error_Q
                    num2 = it

                train_Q_error.append(error_Q.item())  # 记录误差

                with open("结果/损失景观不缩放2/train_loss.txt", 'w') as train_los:
                    train_los.write(str(train_loss))

                with open("结果/损失景观不缩放2/train_Q_error.txt", 'w') as train_ac:
                    train_ac.write(str(train_Q_error))

                if self.useObs:  # 新增两PDE损失项,修改BC损失中Q和S,增加BC损失中Z，删去IC  没有A0
                    loss_BC_Q, loss_BC_Z, loss_obs_Q, loss_obs_Z, loss_obs_S, loss_f_c, loss_f_m, loss_f_s, loss_f_b = \
                        self.sess.run([self.loss_BC_Q, self.loss_BC_Z, self.loss_obs_Q, self.loss_obs_Z,
                                       self.loss_obs_S,
                                       self.loss_f_c, self.loss_f_m, self.loss_f_s, self.loss_f_b], tf_dict)

                else:
                    loss_BC_Q, loss_BC_Z, loss_f_c, loss_f_m, loss_f_s, loss_f_b = \
                        self.sess.run([self.loss_BC_Q,  self.loss_BC_Z, self.loss_f_c, self.loss_f_m,
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
                # self.loss_BC_S_log.append(loss_BC_S)  # 已改
                self.loss_BC_Z_log.append(loss_BC_Z)  # 新增
                # self.loss_IC_h_log.append(loss_IC_h)
                if self.useObs:  # obs新增
                    self.loss_obs_Q_log.append(loss_obs_Q)
                    self.loss_obs_Z_log.append(loss_obs_Z)
                    # self.loss_obs_U_log.append(loss_obs_U)
                    self.loss_obs_S_log.append(loss_obs_S)
                    # self.loss_obs_A0_log.append(loss_obs_A0)
                self.l2_Q_error_log.append(error_Q)  # 已改
                self.l2_Z_error_log.append(error_Z)  # 已改
                # self.l2_U_error_log.append(error_U)  # 新增
                self.l2_S_error_log.append(error_S)  # 新增
                self.l2_A0_error_log.append(error_A0)  # 新增

                # Compute the adaptive constant
                #   BC 已改
                # adaptive_constant_bcs_Q_val = self.sess.run(self.adaptive_constant_bcs_Q, tf_dict)
                # self.adaptive_constant_bcs_Q_val = adaptive_constant_bcs_Q_val * \
                #                                    (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_Q_val
                # self.adpative_constant_bcs_Q_log.append(self.adaptive_constant_bcs_Q_val)
                # # 已改
                # # adaptive_constant_bcs_S_val = self.sess.run(self.adaptive_constant_bcs_S, tf_dict)
                # # self.adaptive_constant_bcs_S_val = adaptive_constant_bcs_S_val * \
                #                                    # (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_S_val
                # # self.adpative_constant_bcs_S_log.append(self.adaptive_constant_bcs_S_val)
                # # 新增BC_Z
                # adaptive_constant_bcs_Z_val = self.sess.run(self.adaptive_constant_bcs_Z, tf_dict)
                # self.adaptive_constant_bcs_Z_val = adaptive_constant_bcs_Z_val * \
                #                                    (1.0 - self.beta) + self.beta * self.adaptive_constant_bcs_Z_val
                # self.adpative_constant_bcs_Z_log.append(self.adaptive_constant_bcs_Z_val)
                # IC删除
                """
                adaptive_constant_ics_h_val = self.sess.run(self.adaptive_constant_ics_h, tf_dict)
                self.adaptive_constant_ics_h_val = adaptive_constant_ics_h_val * \
                                                   (1.0 - self.beta) + self.beta * self.adaptive_constant_ics_h_val
                self.adpative_constant_ics_h_log.append(self.adaptive_constant_ics_h_val)
                """
                # if self.useObs:
                #     adaptive_constant_obs_Q_val = self.sess.run(self.adaptive_constant_obs_Q, tf_dict)
                #     self.adaptive_constant_obs_Q_val = adaptive_constant_obs_Q_val * \
                #                                        (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_Q_val
                #     self.adpative_constant_obs_Q_log.append(self.adaptive_constant_obs_Q_val)
                #
                #     adaptive_constant_obs_Z_val = self.sess.run(self.adaptive_constant_obs_Z, tf_dict)
                #     self.adaptive_constant_obs_Z_val = adaptive_constant_obs_Z_val * \
                #                                        (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_Z_val
                #     self.adpative_constant_obs_Z_log.append(self.adaptive_constant_obs_Z_val)
                #
                #     # adaptive_constant_obs_U_val = self.sess.run(self.adaptive_constant_obs_U, tf_dict)
                #     # self.adaptive_constant_obs_U_val = adaptive_constant_obs_U_val * \
                #     #                                    (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_U_val
                #     # self.adpative_constant_obs_U_log.append(self.adaptive_constant_obs_U_val)
                #
                #     adaptive_constant_obs_S_val = self.sess.run(self.adaptive_constant_obs_S, tf_dict)
                #     self.adaptive_constant_obs_S_val = adaptive_constant_obs_S_val * \
                #                                        (1.0 - self.beta) + self.beta * self.adaptive_constant_obs_S_val
                #     self.adpative_constant_obs_S_log.append(self.adaptive_constant_obs_S_val)

                    # adaptive_constant_obs_A0_val = self.sess.run(self.adaptive_constant_obs_A0, tf_dict)
                    # self.adaptive_constant_obs_A0_val = adaptive_constant_obs_A0_val * \
                    #                                     (
                    #                                                 1.0 - self.beta) + self.beta * self.adaptive_constant_obs_A0_val
                    # self.adpative_constant_obs_A0_log.append(self.adaptive_constant_obs_A0_val)

                if self.useObs:  # BC已改已增加 IC已删
                    print(
                        "constant_bcs_Q_val: {:.3f}, constant_bcs_Z_val: {:.3f}, "
                        "constant_obs_Q_val: {:.3f}, constant_obs_Z_val: {:.3f},  "
                        "constant_obs_S_val: {:.3f}".format(
                            self.adaptive_constant_bcs_Q_val,
                            self.adaptive_constant_bcs_Z_val, self.adaptive_constant_obs_Q_val,
                            self.adaptive_constant_obs_Z_val,
                            self.adaptive_constant_obs_S_val))

                    # 每50轮记录一下  没有A0
                    if it % 50 == 0:
                        with open("结果/损失景观不缩放2/log.txt", "a") as log_file:  # 打开文件进行追加写入
                            log_file.write(
                                f"Epoch: {it}, Loss: {loss_value:.6f}, Learning Rate: {learning_rate:.6f}, Loss_BC_Q: {loss_BC_Q:.6f}, Loss_BC_Z: {loss_BC_Z:.6f}, "
                                f"Loss_obs_Q: {loss_obs_Q:.6f}, Loss_obs_Z: {loss_obs_Z:.6f}, "
                                f"Loss_obs_S: {loss_obs_S:.6f},  Loss_f_c: {loss_f_c:.6f}, "
                                f"Loss_f_m: {loss_f_m:.6f}, Loss_f_s: {loss_f_s:.6f}, Loss_f_b: {loss_f_b:.6f}, "
                                f"Error_Q: {error_Q:.6f}, Error_Z: {error_Z:.6f},  "
                                f"Error_S: {error_S:.6f}, Error_A0: {error_A0:.6f}\n"
                            )

                    # 每10轮打印
                    print(
                        'Loss_BC_Q: %.3e, Loss_BC_Z: %.3e, Loss_obs_Q: %.3e, Loss_obs_Z: %.3e, Loss_obs_S: %.3e, '
                        'Loss_f_c: %.3e, Loss_f_m: %.3e,Loss_f_s: %.3e,Loss_f_b: %.3e, Error Q: %.3e, Error Z: %.3e, '
                        ' Error S: %.3e, Error A0: %.3e '
                        % (
                            loss_BC_Q, loss_BC_Z, loss_obs_Q, loss_obs_Z, loss_obs_S, loss_f_c,
                            loss_f_m, loss_f_s,
                            loss_f_b, error_Q, error_Z,  error_S, error_A0))

                else:
                    print("constant_bcs_Q_val: {:.3f}, constant_bcs_Z_val: {:.3f}".format(
                        self.adaptive_constant_bcs_Q_val,
                        self.adaptive_constant_bcs_Z_val))

                start_time = time.time()
        self.compute_loss_landscape(grid_size=51, scale_range=0.3, feed_dict=tf_dict)  # 使用train中的tf_dict


    def train_bfgs(self):
        tf_dict = {  # self.x_h_IC_tf: self.x_h_IC, self.t_h_IC_tf: self.t_h_IC, self.h_IC_tf: self.h_IC,   # 已删
            self.x_u_BC_tf: self.x_u_BC, self.t_u_BC_tf: self.t_u_BC, self.Q_BC_tf: self.Q_BC,  # 已改
            self.x_h_BC_tf: self.x_h_BC, self.t_h_BC_tf: self.t_h_BC,  # 已改
            self.x_Z_BC_tf: self.x_Z_BC, self.t_Z_BC_tf: self.t_Z_BC, self.Z_BC_tf: self.Z_BC,  # 新增
            self.x_u_obs_tf: self.x_u_obs, self.t_u_obs_tf: self.t_u_obs, self.Q_obs_tf: self.Q_obs,
            self.x_h_obs_tf: self.x_h_obs, self.t_h_obs_tf: self.t_h_obs, self.Z_obs_tf: self.Z_obs,
            self.S_obs_tf: self.S_obs, self.A0_obs_tf: self.A0_obs,
            self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        if self.useObs:  # 改
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_f_c, self.loss_f_m, self.loss_BC_Q,
                                             self.loss_BC_Z, self.loss_obs_Q, self.loss_obs_Z,
                                             self.loss_obs_S],
                                    loss_callback=self.callback_obs)
        else:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_f_c, self.loss_f_m, self.loss_BC_Q,
                                             self.loss_BC_Z],
                                    loss_callback=self.callback)

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
                                      # np.array(self.adpative_constant_bcs_S_log),  # 已改
                                      np.array(self.adpative_constant_bcs_Z_log),  # 新增
                                      # np.array(self.adpative_constant_ics_h_log),
                                      np.array(self.adpative_constant_obs_Q_log),  # 已改
                                      np.array(self.adpative_constant_obs_Z_log),
                                      # np.array(self.adpative_constant_obs_U_log),
                                      np.array(self.adpative_constant_obs_S_log)
                                      # np.array(self.adpative_constant_obs_A0_log)
                                      ])
        else:
            weight_array = np.vstack([np.array(self.adpative_constant_bcs_Q_log),  # 已改
                                      # np.array(self.adpative_constant_bcs_S_log),  # 已改
                                      np.array(self.adpative_constant_bcs_Z_log)  # 新增
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
