"""
获取方程每30h,每断面 参数A,B,hc,Jf,挟沙力级配delta_S1,悬沙级配delta_S,浑水沉速w
一个断面367个时刻 54断面  共 54 * 367 = 19818
获取x_total

顺序没问题
"""
import numpy as np
import pandas as pd

# 读取Excel文件
file_path = r'output77/每12h参数新.xlsx' # 替换为你的文件路径
sheet_name = 0  # 如果有多个sheet，可替换为对应sheet名称或索引

# 读取数据
df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)


# 筛选出有效数据行
# 定义有效数据条件：非空值且包含指定列名称
def is_valid_row(row):
    try:
        # 将所有值转换为浮点数，如果失败则认为是无效行
        [float(val) for val in row if not pd.isna(val)]
        return True
    except ValueError:
        return False


# 应用过滤规则
valid_rows = df.apply(is_valid_row, axis=1)
data = df[valid_rows]

# 添加列名，替换为你需要的列名
data.columns = [
    "Ncs",
    "DistLG(km)",
    "Hav(m)",
    "Uav(m/s)",
    "Area(m2)",
    "BW(m)",
    "QD(m3/s)",
    "ZW(m)",
    "ZBav(m)",
    "Froude",
    "RN",
    "J(%%)",
    "Qmc%",
    "挟沙力级配1",
    "挟沙力级配2",
    "挟沙力级配3",
    "挟沙力级配4",
    "挟沙力级配5",
    "挟沙力级配6",
    "挟沙力级配7",
    "挟沙力级配8",
    "挟沙力级配9",
    "挟沙力级配10",
    "悬沙级配1",
    "悬沙级配2",
    "悬沙级配3",
    "悬沙级配4",
    "悬沙级配5",
    "悬沙级配6",
    "悬沙级配7",
    "悬沙级配8",
    "悬沙级配9",
    "悬沙级配10",
    "床沙级配1",
    "床沙级配2",
    "床沙级配3",
    "床沙级配4",
    "床沙级配5",
    "床沙级配6",
    "床沙级配7",
    "床沙级配8",
    "床沙级配9",
    "床沙级配10",
    "浑水沉速1",
    "浑水沉速2",
    "浑水沉速3",
    "浑水沉速4",
    "浑水沉速5",
    "浑水沉速6",
    "浑水沉速7",
    "浑水沉速8",
    "浑水沉速9",
    "浑水沉速10",
    "恢复饱和系数1",
    "恢复饱和系数2",
    "恢复饱和系数3",
    "恢复饱和系数4",
    "恢复饱和系数5",
    "恢复饱和系数6",
    "恢复饱和系数7",
    "恢复饱和系数8",
    "恢复饱和系数9",
    "恢复饱和系数10",
    "形心水深hc"
]

# 选择需要的列
filtered_data_H = data[["DistLG(km)", "Hav(m)"]]
filtered_data_n = data[["DistLG(km)", "RN"]]
filtered_data_A = data[["DistLG(km)", "Area(m2)"]]
filtered_data_B = data[["DistLG(km)", "BW(m)"]]
filtered_data_hc = data[["DistLG(km)", "形心水深hc"]]
# filtered_data_Jf = data[["DistLG(km)", "水力坡度Jf"]]
# 协沙力级配
filtered_data_delta_S1 = data[
    ["DistLG(km)", "挟沙力级配1", "挟沙力级配2", "挟沙力级配3", "挟沙力级配4", "挟沙力级配5", "挟沙力级配6",
     "挟沙力级配7", "挟沙力级配8", "挟沙力级配9", "挟沙力级配10"]]  #
# 悬沙级配
filtered_data_delta_S = data[
    ["DistLG(km)", "悬沙级配1", "悬沙级配2", "悬沙级配3", "悬沙级配4", "悬沙级配5", "悬沙级配6",
     "悬沙级配7", "悬沙级配8", "悬沙级配9", "悬沙级配10"]]  #
# 床沙级配
filtered_data_delta_bed = data[
    ["DistLG(km)", "床沙级配1", "床沙级配2", "床沙级配3", "床沙级配4", "床沙级配5", "床沙级配6",
     "床沙级配7", "床沙级配8", "床沙级配9", "床沙级配10"]]  #
filtered_data_w = data[
    ["DistLG(km)", "浑水沉速1", "浑水沉速2", "浑水沉速3", "浑水沉速4", "浑水沉速5", "浑水沉速6",
     "浑水沉速7", "浑水沉速8", "浑水沉速9", "浑水沉速10"]]  #
filtered_data_a = data[
    ["DistLG(km)", "恢复饱和系数1", "恢复饱和系数2", "恢复饱和系数3", "恢复饱和系数4", "恢复饱和系数5", "恢复饱和系数6",
     "恢复饱和系数7", "恢复饱和系数8", "恢复饱和系数9", "恢复饱和系数10"]]
# 查看结果
# print(filtered_data_A.head(200).to_string())

# 假设 filtered_data 是你已经处理过的 DataFrame

# 创建一个空的 DataFrame 用于存放结果
result_data_H = pd.DataFrame()
result_data_n = pd.DataFrame()
result_data_A = pd.DataFrame()
result_data_B = pd.DataFrame()
result_data_hc = pd.DataFrame()
# result_data_Jf = pd.DataFrame()
result_data_delta_S1 = pd.DataFrame()
result_data_delta_S = pd.DataFrame()
result_data_delta_bed = pd.DataFrame()
result_data_w = pd.DataFrame()
result_data_a = pd.DataFrame()

# 遍历所有 DistLG(km) 的唯一值
unique_distlg = filtered_data_A['DistLG(km)'].unique()
# print("位置信息：\n", unique_distlg)
# 将每个 DistLG(km) 对应的 Area(m2) 合并成一行
for distlg in unique_distlg:
    # 获取每个 DistLG(km) 对应的所有 Area(m2) 值
    temp_data_H = filtered_data_H[filtered_data_H['DistLG(km)'] == distlg][['DistLG(km)', 'Hav(m)']]
    temp_data_n = filtered_data_n[filtered_data_n['DistLG(km)'] == distlg][['DistLG(km)', 'RN']]
    temp_data_A = filtered_data_A[filtered_data_A['DistLG(km)'] == distlg][['DistLG(km)', 'Area(m2)']]
    temp_data_B = filtered_data_B[filtered_data_B['DistLG(km)'] == distlg][['DistLG(km)', 'BW(m)']]
    temp_data_hc = filtered_data_hc[filtered_data_hc['DistLG(km)'] == distlg][['DistLG(km)', '形心水深hc']]
    # temp_data_Jf = filtered_data_Jf[filtered_data_Jf['DistLG(km)'] == distlg][['DistLG(km)', '水力坡度Jf']]
    temp_data_delta_S1 = filtered_data_delta_S1[filtered_data_delta_S1['DistLG(km)'] == distlg][
        ['DistLG(km)', '挟沙力级配1', '挟沙力级配2',
         '挟沙力级配3', '挟沙力级配4', '挟沙力级配5',
         '挟沙力级配6', '挟沙力级配7', '挟沙力级配8',
         '挟沙力级配9', '挟沙力级配10']]
    temp_data_delta_S = filtered_data_delta_S[filtered_data_delta_S['DistLG(km)'] == distlg][
        ['DistLG(km)', '悬沙级配1', '悬沙级配2',
         '悬沙级配3', '悬沙级配4', '悬沙级配5',
         '悬沙级配6', '悬沙级配7', '悬沙级配8',
         '悬沙级配9', '悬沙级配10']]
    temp_data_delta_bed = filtered_data_delta_bed[filtered_data_delta_bed['DistLG(km)'] == distlg][
        ['DistLG(km)', '床沙级配1', '床沙级配2',
         '床沙级配3', '床沙级配4', '床沙级配5',
         '床沙级配6', '床沙级配7', '床沙级配8',
         '床沙级配9', '床沙级配10']]
    temp_data_w = filtered_data_w[filtered_data_w['DistLG(km)'] == distlg][
        ['DistLG(km)', '浑水沉速1', '浑水沉速2',
         '浑水沉速3', '浑水沉速4', '浑水沉速5',
         '浑水沉速6', '浑水沉速7', '浑水沉速8',
         '浑水沉速9', '浑水沉速10']]
    temp_data_a = filtered_data_a[filtered_data_a['DistLG(km)'] == distlg][
        ['DistLG(km)', '恢复饱和系数1', '恢复饱和系数2',
         '恢复饱和系数3', '恢复饱和系数4', '恢复饱和系数5',
         '恢复饱和系数6', '恢复饱和系数7', '恢复饱和系数8',
         '恢复饱和系数9', '恢复饱和系数10']]

    # 将结果附加到结果 DataFrame 中
    result_data_H = pd.concat([result_data_H, temp_data_H], ignore_index=True)
    result_data_n = pd.concat([result_data_n, temp_data_n], ignore_index=True)
    result_data_A = pd.concat([result_data_A, temp_data_A], ignore_index=True)
    result_data_B = pd.concat([result_data_B, temp_data_B], ignore_index=True)
    result_data_hc = pd.concat([result_data_hc, temp_data_hc], ignore_index=True)
    # result_data_Jf = pd.concat([result_data_Jf, temp_data_Jf], ignore_index=True)
    result_data_delta_S1 = pd.concat([result_data_delta_S1, temp_data_delta_S1], ignore_index=True)
    result_data_delta_S = pd.concat([result_data_delta_S, temp_data_delta_S], ignore_index=True)
    result_data_delta_bed = pd.concat([result_data_delta_bed, temp_data_delta_bed], ignore_index=True)
    result_data_w = pd.concat([result_data_w, temp_data_w], ignore_index=True)
    result_data_a = pd.concat([result_data_a, temp_data_a], ignore_index=True)

# 打印前 100 行
# print("看看")
print(result_data_delta_S.head(200).to_string())

import pandas as pd

# 假设 filtered_data 是你已经处理过的 DataFrame

# 创建一个空的列表来存储结果
H_list = []
n_list = []
A_list = []
B_list = []
hc_list = []
Jf_list = []
delta_S1_list = []
delta_S_list = []
delta_bed_list = []
w_list = []
w8_list = []
a_list = []

# 遍历 DataFrame 中的每一行，将 DistLG(km) 和 Area(m2) 配对
for _, row in result_data_H.iterrows():
    H_list.append([row['DistLG(km)'], row['Hav(m)']])

for _, row in result_data_n.iterrows():
    n_list.append([row['DistLG(km)'], row['RN']])

for _, row in result_data_A.iterrows():
    A_list.append([row['DistLG(km)'], row['Area(m2)']])

for _, row in result_data_B.iterrows():
    B_list.append([row['DistLG(km)'], row['BW(m)']])

for _, row in result_data_hc.iterrows():
    hc_list.append([row['DistLG(km)'], row['形心水深hc']])

# for _, row in result_data_Jf.iterrows():
#     Jf_list.append([row['DistLG(km)'], row['水力坡度Jf']])

for _, row in result_data_delta_S1.iterrows():
    delta_S1_list.append([row['DistLG(km)'], row['挟沙力级配1'], row['挟沙力级配2'], row['挟沙力级配3'],
                          row['挟沙力级配4'], row['挟沙力级配5'], row['挟沙力级配6'], row['挟沙力级配7'],
                          row['挟沙力级配8'], row['挟沙力级配9'], row['挟沙力级配10']])

for _, row in result_data_delta_S.iterrows():
    delta_S_list.append([row['DistLG(km)'], row['悬沙级配1'], row['悬沙级配2'], row['悬沙级配3'],
                         row['悬沙级配4'], row['悬沙级配5'], row['悬沙级配6'], row['悬沙级配7'],
                         row['悬沙级配8'], row['悬沙级配9'], row['悬沙级配10']])

for _, row in result_data_delta_bed.iterrows():
    delta_bed_list.append([row['DistLG(km)'], row['床沙级配1'], row['床沙级配2'], row['床沙级配3'],
                         row['床沙级配4'], row['床沙级配5'], row['床沙级配6'], row['床沙级配7'],
                         row['床沙级配8'], row['床沙级配9'], row['床沙级配10']])

for _, row in result_data_w.iterrows():
    w_list.append([row['DistLG(km)'], row['浑水沉速1'], row['浑水沉速2'], row['浑水沉速3'],
                         row['浑水沉速4'], row['浑水沉速5'], row['浑水沉速6'], row['浑水沉速7'],
                         row['浑水沉速8'], row['浑水沉速9'], row['浑水沉速10']])

for _, row in result_data_w.iterrows():
    w8_list.append([row['DistLG(km)'], row['浑水沉速3'],
                         row['浑水沉速4'], row['浑水沉速5'], row['浑水沉速6'], row['浑水沉速7'],
                         row['浑水沉速8'], row['浑水沉速9'], row['浑水沉速10']])

for _, row in result_data_a.iterrows():
    a_list.append([row['DistLG(km)'], row['恢复饱和系数1'], row['恢复饱和系数2'], row['恢复饱和系数3'],
                         row['恢复饱和系数4'], row['恢复饱和系数5'], row['恢复饱和系数6'], row['恢复饱和系数7'],
                         row['恢复饱和系数8'], row['恢复饱和系数9'], row['恢复饱和系数10']])
# 打印前 10 个结果
# print("悬沙级配\n",delta_S_list[:30])
# print("面积\n",A_list[:300])
# [[129.67, 0.04, 0.07, 0.11, 0.14, 0.19, 0.22, 0.14, 0.07, 0.01, 0.0],
# [129.67, 0.04, 0.07, 0.1, 0.13, 0.19, 0.23, 0.15, 0.07, 0.01, 0.0]...]


H_columns = [row[1:2] for row in H_list]
n_columns = [row[1:2] for row in n_list]
A_columns = [row[1:2] for row in A_list]
B_columns = [row[1:2] for row in B_list]
hc_columns = [row[1:2] for row in hc_list]
# Jf_columns = [row[1:2] for row in Jf_list]


H = np.array(H_columns[:19818])
print(H)
n = np.array(n_columns[:19818])
A = np.array(A_columns[:19818])
print("A.shape:\n",A.shape)
B = np.array(B_columns[:19818])
hc = np.array(hc_columns[:19818])
# Jf = np.array(Jf_columns)
print(n)



last_10S_columns = [row[1:] for row in delta_S_list]
print()
last_10S_1_columns = [row[1:] for row in delta_S1_list]   # 分组挟沙力
last_10_bed_columns = [row[1:] for row in delta_bed_list]   # 床沙级配
last_10w_columns = [row[1:] for row in w_list]  # 分组浑水沉速
last_8w_columns = [row[1:] for row in w8_list]
last_10a_columns = [row[1:] for row in a_list]
print("last_10a_columns.shape:\n",len(last_10a_columns[:19818]))

bed = np.array(last_10_bed_columns[:19818])
delta_S_1 = np.array(last_10S_1_columns[:19818])
delta_S = np.array(last_10S_columns[:19818])

w = np.array(last_10w_columns[:19818])
w8 = np.array(last_8w_columns[:19818])
print("w8.shape:\n",w8.shape)

a = np.array(last_10a_columns[:19818])
print("浑水沉速w:",w8)
print("恢复饱和系数a：\n", a)
# print(last_10S_columns[:30])
# [[0.04, 0.07, 0.11, 0.14, 0.19, 0.22, 0.14, 0.07, 0.01, 0.0],
# [0.04, 0.07, 0.1, 0.13, 0.19, 0.23, 0.15, 0.07, 0.01, 0.0]...]

x_total = [row[0] for row in delta_S_list]
x_total = x_total[:19818]

x_total = [[x] for x in x_total]
x_total = np.array(x_total)
print("x_total.shape:\n",x_total.shape)
# x_true = [[value] for value in x_total]
# x_true = np.array(x_true)
# # print("空间序列x_total\n" , x_true)
#
# X0 = 1
# x_total = [round(value / X0, 4) for value in x_total]
# x_total = [[value] for value in x_total]
# print("空间序列x_total\n",x_total)

print("x_total.len:\n" ,len(x_total))  # 7884
