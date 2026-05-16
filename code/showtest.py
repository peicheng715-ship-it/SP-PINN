import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

def plot_smooth_field(df, true_col, pred_col, title, save_dir):

    x = df["x"].values
    t = df["t"].values

    true_val = df[true_col].values
    pred_val = df[pred_col].values

    # =====================================================
    # 构造规则网格（关键）
    # =====================================================
    xi = np.linspace(x.min(), x.max(), 200)
    ti = np.linspace(t.min(), t.max(), 200)

    X, T = np.meshgrid(xi, ti)

    # =====================================================
    # 插值（平滑关键）
    # =====================================================
    true_grid = griddata((x, t), true_val, (X, T), method='cubic')
    pred_grid = griddata((x, t), pred_val, (X, T), method='cubic')

    error_grid = np.abs(true_grid - pred_grid)

    # =====================================================
    # 色标统一（关键）
    # =====================================================
    vmin = np.nanmin([true_grid, pred_grid])
    vmax = np.nanmax([true_grid, pred_grid])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 真值 ----
    im0 = axes[0].imshow(true_grid, extent=[x.min(), x.max(), t.min(), t.max()],
                         origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title} True")
    fig.colorbar(im0, ax=axes[0])

    # ---- 预测 ----
    im1 = axes[1].imshow(pred_grid, extent=[x.min(), x.max(), t.min(), t.max()],
                         origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title} Prediction")
    fig.colorbar(im1, ax=axes[1])

    # ---- 误差 ----
    im2 = axes[2].imshow(error_grid, extent=[x.min(), x.max(), t.min(), t.max()],
                         origin='lower', aspect='auto', cmap='inferno')
    axes[2].set_title(f"{title} Error")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{title}.png"), dpi=300)
    plt.show()


df_all = pd.read_excel("/home/cp/overbank3/结果/插值/插值测试结果.xlsx")

# Q
plot_smooth_field(df_all, "Q_true", "Q_pred", "Q", "结果/热力图")

# Z
plot_smooth_field(df_all, "Z_true", "Z_pred", "Z", "结果/热力图")

# S（注意统一尺度！）
plot_smooth_field(df_all, "S_true", "S_pred", "S", "结果/热力图")