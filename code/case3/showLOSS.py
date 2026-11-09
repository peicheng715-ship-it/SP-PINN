import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load('D:/2017/结果/结果/无输入/不缩放/网络512/loss_landscape_data.npz')
alphas, betas, losses = data['alphas'], data['betas'], data['losses']
losses = np.log(losses + 1e-6)  # 加小值防 log0

# 3D表面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
A, B = np.meshgrid(alphas, betas)
ax.plot_surface(A, B, losses, cmap='viridis')
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
# 然后 plot_surface 或 contourf
ax.set_zlabel('Log(Loss)')  # 标注
plt.show()
# plt.savefig('loss_landscape_3d.png')

# 2D等高线
plt.figure()
plt.contourf(A, B, losses, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.show()
# plt.savefig('loss_landscape_2d.png')