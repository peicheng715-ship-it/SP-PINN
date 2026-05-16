import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# 基础设置：你可以调这个
# n_insert_x = 1  表示每两个相邻断面之间插入 1 个新断面（中点）
# n_insert_t = 1  表示每两个相邻时刻之间插入 1 个新时刻（中点）
# ============================================================
n_insert_x = 1
n_insert_t = 1
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# 基础设置：你可以调这个
# n_insert_x = 1  表示每两个相邻断面之间插入 1 个新断面（中点）
# n_insert_t = 1  表示每两个相邻时刻之间插入 1 个新时刻（中点）
# ============================================================
n_insert_x = 1
n_insert_t = 1

# ============================================================
# 工具函数
# ============================================================
def refine_axis(axis, n_insert=1):
    """
    在相邻点之间均匀插值插入 n_insert 个点。
    例如 n_insert=1 -> 插入中点
    """
    axis = np.asarray(axis, dtype=float)
    axis = np.sort(np.unique(axis))

    if n_insert <= 0:
        return axis.copy()

    refined = [axis[0]]
    for a, b in zip(axis[:-1], axis[1:]):
        inner = np.linspace(a, b, n_insert + 2)[1:-1]
        refined.extend(inner.tolist())
        refined.append(b)
    return np.array(refined, dtype=float)


def read_wide_table(path):
    """
    读取这种格式：
    - 第一列：t
    - 第二列开始：不同 x 位置对应的值
    返回：
    x: (Nx,)
    t: (Nt,)
    F: (Nt, Nx)
    """
    df = pd.read_excel(path).dropna(how="all").reset_index(drop=True)

    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    x = np.array([float(c) for c in df.columns[1:]], dtype=float)
    F = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if np.isnan(t).any():
        raise ValueError(f"{path} 第一列时间存在无法转成数值的项。")
    if np.isnan(F).any():
        print(f"警告：{path} 中存在 NaN，后续插值将自动跳过这些空值区域。")

    return x, t, F


def read_flat_series(path, usecol):
    """
    读取 table.xlsx 这种按行展开的一维列数据
    返回：
    x_flat: (N,)
    v_flat: (N,)
    """
    df = pd.read_excel(path, usecols=[0, usecol], skiprows=2)
    df = df.dropna()

    x_flat = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    v_flat = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

    mask = ~(np.isnan(x_flat) | np.isnan(v_flat))
    return x_flat[mask], v_flat[mask]


def flat_to_grid(x_flat, t_flat, v_flat, x_base, t_base):
    """
    把按 (x,t) 展开的平铺数据还原成规则网格，返回 shape = (Nt, Nx)
    这里假设你的顺序是：
        断面1所有时刻 -> 断面2所有时刻 -> ...
    即 x-major, t-minor
    """
    x_flat = np.asarray(x_flat, dtype=float).ravel()
    t_flat = np.asarray(t_flat, dtype=float).ravel()
    v_flat = np.asarray(v_flat, dtype=float).ravel()

    x_base = np.asarray(np.sort(np.unique(x_base)), dtype=float)
    t_base = np.asarray(np.sort(np.unique(t_base)), dtype=float)

    # 为了避免浮点比较误差，统一做一个轻微 round
    x_key = np.round(x_flat, 10)
    t_key = np.round(t_flat, 10)
    x_base_key = np.round(x_base, 10)
    t_base_key = np.round(t_base, 10)

    x_map = {val: i for i, val in enumerate(x_base_key)}
    t_map = {val: i for i, val in enumerate(t_base_key)}

    ix = np.array([x_map[v] for v in x_key], dtype=int)
    it = np.array([t_map[v] for v in t_key], dtype=int)

    grid = np.full((len(t_base), len(x_base)), np.nan, dtype=float)
    grid[it, ix] = v_flat
    return grid


def interp_on_dense_grid(x_base, t_base, F_base, x_dense, t_dense, method="linear"):
    """
    在规则网格上做二维插值。
    输入 F_base shape = (Nt, Nx)
    输出 shape = (Nt_dense, Nx_dense)
    """
    x_base = np.asarray(x_base, dtype=float)
    t_base = np.asarray(t_base, dtype=float)
    F_base = np.asarray(F_base, dtype=float)

    # 保证单调递增
    ix = np.argsort(x_base)
    it = np.argsort(t_base)
    x_base = x_base[ix]
    t_base = t_base[it]
    F_base = F_base[np.ix_(it, ix)]

    interp = RegularGridInterpolator(
        (t_base, x_base),
        F_base,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    Tn, Xn = np.meshgrid(t_dense, x_dense, indexing="ij")
    pts = np.column_stack([Tn.ravel(), Xn.ravel()])
    F_dense = interp(pts).reshape(len(t_dense), len(x_dense))
    return F_dense


def compute_delta_s(q_scaled):
    """
    根据缩放后的 Q 重新构造 delta_S
    这里使用与你原代码一致的分段规则
    """
    q = np.asarray(q_scaled, dtype=float).ravel()

    # 你的原始阈值（注意：这是缩放后的 Q）
    q1 = 100 / 500
    q2 = 1000 / 500
    q3 = 8000 / 500

    # 为避免极少数数值插值略超界，先截断到合理范围
    q = np.clip(q, 0.0, q3)

    delta_S_100 = np.array([0.4, 0.3, 0.2, 0.08, 0.01, 0.01, 0.01], dtype=float)
    delta_S_1000 = np.array([0.5, 0.2, 0.2, 0.08, 0.01, 0.01, 0.01], dtype=float)
    delta_S_5000 = np.array([0.55, 0.18, 0.15, 0.08, 0.02, 0.01, 0.01], dtype=float)

    out = np.zeros((len(q), 7), dtype=float)
    m1 = (q >= 0) & (q < q1)
    m2 = (q >= q1) & (q < q2)
    m3 = (q >= q2) & (q <= q3)

    out[m1] = delta_S_100
    out[m2] = delta_S_1000
    out[m3] = delta_S_5000

    return out


# ============================================================
# 1) 读取你的原始数据
# ============================================================

# ---- x / t 基础网格来源 ----
# x 来自 table.xlsx 第 1 列
df_x = pd.read_excel(r"data/table.xlsx", usecols=[0], skiprows=2).dropna()
x_flat = pd.to_numeric(df_x.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
x_flat = x_flat[~np.isnan(x_flat)]

# t 来自 sediment.xlsx 第一列
df_t = pd.read_excel(r"data/sediment.xlsx")
t_base = pd.to_numeric(df_t.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
t_base = t_base[~np.isnan(t_base)]

# 提取唯一 x，作为空间基准
x_base = np.sort(np.unique(x_flat))
t_base = np.sort(np.unique(t_base))

nx = len(x_base)
nt = len(t_base)

print("原始 x 点数 nx =", nx)
print("原始 t 点数 nt =", nt)

# 你要求的顺序：断面1所有时间点 -> 断面2所有时间点 -> ...
# 所以这里构造 t_flat 为 x-major 对应顺序
t_flat = np.tile(t_base, nx)

if len(x_flat) != len(t_flat):
    raise ValueError(
        f"x_flat 长度 {len(x_flat)} 与 t_flat 长度 {len(t_flat)} 不一致，"
        f"请检查 table.xlsx 与时间序列是否对应。"
    )

# ---- Q / A / B：平铺的一维数据 ----
x_q, Q_flat = read_flat_series(r"data/table.xlsx", usecol=2)
x_a, A_flat = read_flat_series(r"data/table.xlsx", usecol=5)
x_b, B_flat = read_flat_series(r"data/table.xlsx", usecol=6)

# 如果它们的 x 顺序和 x_flat 一致，这里就直接复用 x_flat
# 否则你可以把下面三行改成对应读取结果
Q_flat = Q_flat / 500.0  # 保持与你原始代码一致的缩放
A_flat = A_flat.copy()
B_flat = B_flat.copy()

if not (len(Q_flat) == len(x_flat) == len(A_flat) == len(B_flat)):
    raise ValueError("Q / A / B 的长度与 x_flat 不一致，请检查 Excel 列。")

# ---- H / S / W：宽表数据（第一列 t，后面各列 x）----
x_S_wide, t_S_wide, S_wide = read_wide_table(r"data/sediment.xlsx")
x_H_wide, t_H_wide, H_wide = read_wide_table(r"data/depth.xlsx")
x_W_wide, t_W_wide, W_wide = read_wide_table(r"data/fall velocity.xlsx")

# ============================================================
# 2) 把平铺的一维量恢复成规则二维网格
#    统一成 shape = (Nt, Nx)
# ============================================================
Q_grid = flat_to_grid(x_flat, t_flat, Q_flat, x_base, t_base)
A_grid = flat_to_grid(x_flat, t_flat, A_flat, x_base, t_base)
B_grid = flat_to_grid(x_flat, t_flat, B_flat, x_base, t_base)

# 宽表数据本来就是规则网格，这里直接对齐
# 如果 x/t 顺序和 base 不完全一致，插值时会自动按 base 重建
S_grid = S_wide.copy()
H_grid = H_wide.copy()
W_grid = W_wide.copy()

# 检查宽表网格是否与 base 一致，不一致也没关系，只要范围对应即可
if not np.allclose(np.sort(np.unique(x_S_wide)), x_base):
    print("警告：sediment.xlsx 的 x 位置与 table.xlsx 不完全一致，仍然继续插值。")
if not np.allclose(np.sort(np.unique(t_S_wide)), t_base):
    print("警告：sediment.xlsx 的 t 位置与 table.xlsx 不完全一致，仍然继续插值。")

# ============================================================
# 3) 构造新的密集网格
# ============================================================
x_dense = refine_axis(x_base, n_insert=n_insert_x)
t_dense = refine_axis(t_base, n_insert=n_insert_t)

print("x_dense 点数 =", len(x_dense))
print("t_dense 点数 =", len(t_dense))

# ============================================================
# 4) 对 Q / H / S / W / A / B 做二维插值
# ============================================================
Q_dense = interp_on_dense_grid(x_base, t_base, Q_grid, x_dense, t_dense, method="linear")
H_dense = interp_on_dense_grid(x_base, t_base, H_grid, x_dense, t_dense, method="linear")
S_dense = interp_on_dense_grid(x_base, t_base, S_grid, x_dense, t_dense, method="linear")
W_dense = interp_on_dense_grid(x_base, t_base, W_grid, x_dense, t_dense, method="linear")
A_dense = interp_on_dense_grid(x_base, t_base, A_grid, x_dense, t_dense, method="linear")
B_dense = interp_on_dense_grid(x_base, t_base, B_grid, x_dense, t_dense, method="linear")

# ============================================================
# 5) 常数/分组参数：直接复制，不插值
# ============================================================
n_const = 0.035
S1_const = 0.3
a_const = np.array([0.01, 0.01, 0.007, 0.01, 0.01, 0.01, 0.011], dtype=float)

# 网格总点数
N_dense = len(x_dense) * len(t_dense)

# 常数复制到每个点
n_dense = np.full((N_dense, 1), n_const, dtype=float)
S1_dense = np.full((N_dense, 1), S1_const, dtype=float)
a_dense = np.tile(a_const.reshape(1, -1), (N_dense, 1))  # (N, 7)

# delta_S 由插值后的 Q 重新计算
# 注意：这里用的是与你原始代码一致的 Q 缩放值
Q_dense_flat = Q_dense.T.reshape(-1, 1)  # 先转成 x-major 再展平
delta_S_dense = compute_delta_s(Q_dense_flat.ravel())  # (N, 7)

# ============================================================
# 6) 组织成你要的 x-major 顺序：
#    断面1所有时间点 -> 断面2所有时间点 -> ...
# ============================================================
# 先把 dense 数据转成 x-major 排列
# 当前插值结果是 (Nt, Nx)，所以先转置成 (Nx, Nt)
Q_xmajor = Q_dense.T
H_xmajor = H_dense.T
S_xmajor = S_dense.T
W_xmajor = W_dense.T
A_xmajor = A_dense.T
B_xmajor = B_dense.T

X_out, T_out = np.meshgrid(x_dense, t_dense, indexing="ij")  # shape = (Nx, Nt)

# 各列展开
out_df = pd.DataFrame({
    "x": X_out.ravel(),
    "t": T_out.ravel(),
    "Q_interp": Q_xmajor.ravel(),
    "H_interp": H_xmajor.ravel(),
    "S_interp": S_xmajor.ravel(),
    "W_interp": W_xmajor.ravel(),
    "A_interp": A_xmajor.ravel(),
    "B_interp": B_xmajor.ravel(),
    "n": np.full(X_out.size, n_const, dtype=float),
    "S1": np.full(X_out.size, S1_const, dtype=float),
})

# a 的 7 个分量
for i in range(7):
    out_df[f"a{i+1}"] = np.full(X_out.size, a_const[i], dtype=float)

# delta_S 的 7 个分量
for i in range(7):
    out_df[f"delta_S{i+1}"] = delta_S_dense[:, i]

# ============================================================
# 7) 生成“测试点”——只保留新插入的点
#    原始训练点定义为：x 在原始 x_base 且 t 在原始 t_base
# ============================================================
tol = 1e-12
x_is_orig = np.isclose(X_out[..., None], x_base[None, None, :], atol=tol, rtol=0).any(axis=-1)
t_is_orig = np.isclose(T_out[..., None], t_base[None, None, :], atol=tol, rtol=0).any(axis=-1)

train_mask = x_is_orig & t_is_orig
test_mask = ~train_mask

test_df = out_df.loc[test_mask.ravel()].reset_index(drop=True)

# ============================================================
# 8) 写入同一个 Excel 文件
# ============================================================
save_path = r"data/PINN_interpolated_dataset.xlsx"

with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
    out_df.to_excel(writer, sheet_name="dense_all", index=False)
    test_df.to_excel(writer, sheet_name="test_only", index=False)

print(f"已生成：{save_path}")
print("dense_all 行数：", len(out_df))
print("test_only 行数：", len(test_df))
print("列名：", list(out_df.columns))



# ============================================================
# 工具函数
# ============================================================
def refine_axis(axis, n_insert=1):
    """
    在相邻点之间均匀插值插入 n_insert 个点。
    例如 n_insert=1 -> 插入中点
    """
    axis = np.asarray(axis, dtype=float)
    axis = np.sort(np.unique(axis))

    if n_insert <= 0:
        return axis.copy()

    refined = [axis[0]]
    for a, b in zip(axis[:-1], axis[1:]):
        inner = np.linspace(a, b, n_insert + 2)[1:-1]
        refined.extend(inner.tolist())
        refined.append(b)
    return np.array(refined, dtype=float)


def read_wide_table(path):
    """
    读取这种格式：
    - 第一列：t
    - 第二列开始：不同 x 位置对应的值
    返回：
    x: (Nx,)
    t: (Nt,)
    F: (Nt, Nx)
    """
    df = pd.read_excel(path).dropna(how="all").reset_index(drop=True)

    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    x = np.array([float(c) for c in df.columns[1:]], dtype=float)
    F = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if np.isnan(t).any():
        raise ValueError(f"{path} 第一列时间存在无法转成数值的项。")
    if np.isnan(F).any():
        print(f"警告：{path} 中存在 NaN，后续插值将自动跳过这些空值区域。")

    return x, t, F


def read_flat_series(path, usecol):
    """
    读取 table.xlsx 这种按行展开的一维列数据
    返回：
    x_flat: (N,)
    v_flat: (N,)
    """
    df = pd.read_excel(path, usecols=[0, usecol], skiprows=2)
    df = df.dropna()

    x_flat = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    v_flat = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

    mask = ~(np.isnan(x_flat) | np.isnan(v_flat))
    return x_flat[mask], v_flat[mask]


def flat_to_grid(x_flat, t_flat, v_flat, x_base, t_base):
    """
    把按 (x,t) 展开的平铺数据还原成规则网格，返回 shape = (Nt, Nx)
    这里假设你的顺序是：
        断面1所有时刻 -> 断面2所有时刻 -> ...
    即 x-major, t-minor
    """
    x_flat = np.asarray(x_flat, dtype=float).ravel()
    t_flat = np.asarray(t_flat, dtype=float).ravel()
    v_flat = np.asarray(v_flat, dtype=float).ravel()

    x_base = np.asarray(np.sort(np.unique(x_base)), dtype=float)
    t_base = np.asarray(np.sort(np.unique(t_base)), dtype=float)

    # 为了避免浮点比较误差，统一做一个轻微 round
    x_key = np.round(x_flat, 10)
    t_key = np.round(t_flat, 10)
    x_base_key = np.round(x_base, 10)
    t_base_key = np.round(t_base, 10)

    x_map = {val: i for i, val in enumerate(x_base_key)}
    t_map = {val: i for i, val in enumerate(t_base_key)}

    ix = np.array([x_map[v] for v in x_key], dtype=int)
    it = np.array([t_map[v] for v in t_key], dtype=int)

    grid = np.full((len(t_base), len(x_base)), np.nan, dtype=float)
    grid[it, ix] = v_flat
    return grid


def interp_on_dense_grid(x_base, t_base, F_base, x_dense, t_dense, method="linear"):
    """
    在规则网格上做二维插值。
    输入 F_base shape = (Nt, Nx)
    输出 shape = (Nt_dense, Nx_dense)
    """
    x_base = np.asarray(x_base, dtype=float)
    t_base = np.asarray(t_base, dtype=float)
    F_base = np.asarray(F_base, dtype=float)

    # 保证单调递增
    ix = np.argsort(x_base)
    it = np.argsort(t_base)
    x_base = x_base[ix]
    t_base = t_base[it]
    F_base = F_base[np.ix_(it, ix)]

    interp = RegularGridInterpolator(
        (t_base, x_base),
        F_base,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    Tn, Xn = np.meshgrid(t_dense, x_dense, indexing="ij")
    pts = np.column_stack([Tn.ravel(), Xn.ravel()])
    F_dense = interp(pts).reshape(len(t_dense), len(x_dense))
    return F_dense


def compute_delta_s(q_scaled):
    """
    根据缩放后的 Q 重新构造 delta_S
    这里使用与你原代码一致的分段规则
    """
    q = np.asarray(q_scaled, dtype=float).ravel()

    # 你的原始阈值（注意：这是缩放后的 Q）
    q1 = 100 / 500
    q2 = 1000 / 500
    q3 = 8000 / 500

    # 为避免极少数数值插值略超界，先截断到合理范围
    q = np.clip(q, 0.0, q3)

    delta_S_100 = np.array([0.4, 0.3, 0.2, 0.08, 0.01, 0.01, 0.01], dtype=float)
    delta_S_1000 = np.array([0.5, 0.2, 0.2, 0.08, 0.01, 0.01, 0.01], dtype=float)
    delta_S_5000 = np.array([0.55, 0.18, 0.15, 0.08, 0.02, 0.01, 0.01], dtype=float)

    out = np.zeros((len(q), 7), dtype=float)
    m1 = (q >= 0) & (q < q1)
    m2 = (q >= q1) & (q < q2)
    m3 = (q >= q2) & (q <= q3)

    out[m1] = delta_S_100
    out[m2] = delta_S_1000
    out[m3] = delta_S_5000

    return out


# ============================================================
# 1) 读取你的原始数据
# ============================================================

# ---- x / t 基础网格来源 ----
# x 来自 table.xlsx 第 1 列
df_x = pd.read_excel(r"data/table.xlsx", usecols=[0], skiprows=2).dropna()
x_flat = pd.to_numeric(df_x.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
x_flat = x_flat[~np.isnan(x_flat)]

# t 来自 sediment.xlsx 第一列
df_t = pd.read_excel(r"data/sediment.xlsx")
t_base = pd.to_numeric(df_t.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
t_base = t_base[~np.isnan(t_base)]

# 提取唯一 x，作为空间基准
x_base = np.sort(np.unique(x_flat))
t_base = np.sort(np.unique(t_base))

nx = len(x_base)
nt = len(t_base)

print("原始 x 点数 nx =", nx)
print("原始 t 点数 nt =", nt)

# 你要求的顺序：断面1所有时间点 -> 断面2所有时间点 -> ...
# 所以这里构造 t_flat 为 x-major 对应顺序
t_flat = np.tile(t_base, nx)

if len(x_flat) != len(t_flat):
    raise ValueError(
        f"x_flat 长度 {len(x_flat)} 与 t_flat 长度 {len(t_flat)} 不一致，"
        f"请检查 table.xlsx 与时间序列是否对应。"
    )

# ---- Q / A / B：平铺的一维数据 ----
x_q, Q_flat = read_flat_series(r"data/table.xlsx", usecol=2)
x_a, A_flat = read_flat_series(r"data/table.xlsx", usecol=5)
x_b, B_flat = read_flat_series(r"data/table.xlsx", usecol=6)

# 如果它们的 x 顺序和 x_flat 一致，这里就直接复用 x_flat
# 否则你可以把下面三行改成对应读取结果
Q_flat = Q_flat / 500.0  # 保持与你原始代码一致的缩放
A_flat = A_flat.copy()
B_flat = B_flat.copy()

if not (len(Q_flat) == len(x_flat) == len(A_flat) == len(B_flat)):
    raise ValueError("Q / A / B 的长度与 x_flat 不一致，请检查 Excel 列。")

# ---- H / S / W：宽表数据（第一列 t，后面各列 x）----
x_S_wide, t_S_wide, S_wide = read_wide_table(r"data/sediment.xlsx")
x_H_wide, t_H_wide, H_wide = read_wide_table(r"data/depth.xlsx")
x_W_wide, t_W_wide, W_wide = read_wide_table(r"data/fall velocity.xlsx")

# ============================================================
# 2) 把平铺的一维量恢复成规则二维网格
#    统一成 shape = (Nt, Nx)
# ============================================================
Q_grid = flat_to_grid(x_flat, t_flat, Q_flat, x_base, t_base)
A_grid = flat_to_grid(x_flat, t_flat, A_flat, x_base, t_base)
B_grid = flat_to_grid(x_flat, t_flat, B_flat, x_base, t_base)

# 宽表数据本来就是规则网格，这里直接对齐
# 如果 x/t 顺序和 base 不完全一致，插值时会自动按 base 重建
S_grid = S_wide.copy()
H_grid = H_wide.copy()
W_grid = W_wide.copy()

# 检查宽表网格是否与 base 一致，不一致也没关系，只要范围对应即可
if not np.allclose(np.sort(np.unique(x_S_wide)), x_base):
    print("警告：sediment.xlsx 的 x 位置与 table.xlsx 不完全一致，仍然继续插值。")
if not np.allclose(np.sort(np.unique(t_S_wide)), t_base):
    print("警告：sediment.xlsx 的 t 位置与 table.xlsx 不完全一致，仍然继续插值。")

# ============================================================
# 3) 构造新的密集网格
# ============================================================
x_dense = refine_axis(x_base, n_insert=n_insert_x)
t_dense = refine_axis(t_base, n_insert=n_insert_t)

print("x_dense 点数 =", len(x_dense))
print("t_dense 点数 =", len(t_dense))

# ============================================================
# 4) 对 Q / H / S / W / A / B 做二维插值
# ============================================================
Q_dense = interp_on_dense_grid(x_base, t_base, Q_grid, x_dense, t_dense, method="linear")
H_dense = interp_on_dense_grid(x_base, t_base, H_grid, x_dense, t_dense, method="linear")
S_dense = interp_on_dense_grid(x_base, t_base, S_grid, x_dense, t_dense, method="linear")
W_dense = interp_on_dense_grid(x_base, t_base, W_grid, x_dense, t_dense, method="linear")
A_dense = interp_on_dense_grid(x_base, t_base, A_grid, x_dense, t_dense, method="linear")
B_dense = interp_on_dense_grid(x_base, t_base, B_grid, x_dense, t_dense, method="linear")

# ============================================================
# 5) 常数/分组参数：直接复制，不插值
# ============================================================
n_const = 0.035
S1_const = 0.3
a_const = np.array([0.01, 0.01, 0.007, 0.01, 0.01, 0.01, 0.011], dtype=float)

# 网格总点数
N_dense = len(x_dense) * len(t_dense)

# 常数复制到每个点
n_dense = np.full((N_dense, 1), n_const, dtype=float)
S1_dense = np.full((N_dense, 1), S1_const, dtype=float)
a_dense = np.tile(a_const.reshape(1, -1), (N_dense, 1))  # (N, 7)

# delta_S 由插值后的 Q 重新计算
# 注意：这里用的是与你原始代码一致的 Q 缩放值
Q_dense_flat = Q_dense.T.reshape(-1, 1)  # 先转成 x-major 再展平
delta_S_dense = compute_delta_s(Q_dense_flat.ravel())  # (N, 7)

# ============================================================
# 6) 组织成你要的 x-major 顺序：
#    断面1所有时间点 -> 断面2所有时间点 -> ...
# ============================================================
# 先把 dense 数据转成 x-major 排列
# 当前插值结果是 (Nt, Nx)，所以先转置成 (Nx, Nt)
Q_xmajor = Q_dense.T
H_xmajor = H_dense.T
S_xmajor = S_dense.T
W_xmajor = W_dense.T
A_xmajor = A_dense.T
B_xmajor = B_dense.T

X_out, T_out = np.meshgrid(x_dense, t_dense, indexing="ij")  # shape = (Nx, Nt)

# 各列展开
out_df = pd.DataFrame({
    "x": X_out.ravel(),
    "t": T_out.ravel(),
    "Q_interp": Q_xmajor.ravel(),
    "H_interp": H_xmajor.ravel(),
    "S_interp": S_xmajor.ravel(),
    "W_interp": W_xmajor.ravel(),
    "A_interp": A_xmajor.ravel(),
    "B_interp": B_xmajor.ravel(),
    "n": np.full(X_out.size, n_const, dtype=float),
    "S1": np.full(X_out.size, S1_const, dtype=float),
})

# a 的 7 个分量
for i in range(7):
    out_df[f"a{i+1}"] = np.full(X_out.size, a_const[i], dtype=float)

# delta_S 的 7 个分量
for i in range(7):
    out_df[f"delta_S{i+1}"] = delta_S_dense[:, i]

# ============================================================
# 7) 生成“测试点”——只保留新插入的点
#    原始训练点定义为：x 在原始 x_base 且 t 在原始 t_base
# ============================================================
tol = 1e-12
x_is_orig = np.isclose(X_out[..., None], x_base[None, None, :], atol=tol, rtol=0).any(axis=-1)
t_is_orig = np.isclose(T_out[..., None], t_base[None, None, :], atol=tol, rtol=0).any(axis=-1)

train_mask = x_is_orig & t_is_orig
test_mask = ~train_mask

test_df = out_df.loc[test_mask.ravel()].reset_index(drop=True)

# ============================================================
# 8) 写入同一个 Excel 文件
# ============================================================
save_path = r"data/PINN_interpolated_dataset.xlsx"

with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
    out_df.to_excel(writer, sheet_name="dense_all", index=False)
    test_df.to_excel(writer, sheet_name="test_only", index=False)

print(f"已生成：{save_path}")
print("dense_all 行数：", len(out_df))
print("test_only 行数：", len(test_df))
print("列名：", list(out_df.columns))


