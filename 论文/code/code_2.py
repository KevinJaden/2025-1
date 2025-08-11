import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 坐标轴负号的处理
plt.rcParams['axes.unicode_minus'] = False
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

sns.set_style("whitegrid")

df = pd.read_excel('附件1.xlsx', sheet_name='性能数据表')
df['催化剂组合编号'] = df['催化剂组合编号'].ffill()
df['催化剂组合'] = df['催化剂组合'].ffill()
df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')
data_rows = []
current_id = None
current_info = {}
for idx, row in df.iterrows():
    if pd.notna(row['催化剂组合编号']) and 'mg' in str(row['催化剂组合']):
        current_id = row['催化剂组合编号']
        combo_str = row['催化剂组合']
        co_match = re.search(r'(\d+\.?\d*)wt%Co/SiO2', combo_str)
        co_loading = float(co_match.group(1)) if co_match else np.nan
        co_sio2_match = re.search(r'(\d+)mg\s+[^\s]+Co/SiO2', combo_str)
        co_sio2_mass = float(co_sio2_match.group(1)) if co_sio2_match else np.nan
        hap_match = re.search(r'(\d+)mg\s+HAP', combo_str)
        hap_mass = float(hap_match.group(1)) if hap_match else 0
        has_hap = 1 if hap_match else 0
        ethanol_match = re.search(r'乙醇浓度\s*(\d+\.?\d*)\s*ml/min', combo_str)
        ethanol_conc = float(ethanol_match.group(1)) if ethanol_match else np.nan
        total_mass = co_sio2_mass + hap_mass
        mass_ratio = co_sio2_mass / hap_mass if hap_mass != 0 else np.nan
        current_info = {
            '编号': current_id,
            'Co负载量(wt%)': co_loading,
            'Co/SiO2质量(mg)': co_sio2_mass,
            'HAP质量(mg)': hap_mass,
            '总质量(mg)': total_mass,
            '质量比(Co/SiO2:HAP)': mass_ratio,
            '乙醇浓度(ml/min)': ethanol_conc,
            '装料方式': 'I' if str(current_id).startswith('A') else 'II',
            '有无HAP': has_hap
        }
        if pd.notna(row.get('温度')) and np.isfinite(row['温度']):
            record = current_info.copy()
            record['温度'] = row['温度']
            record['乙醇转化率(%)'] = row['乙醇转化率(%)']
            record['C4烯烃选择性(%)'] = row['C4烯烃选择性(%)']
            data_rows.append(record)
df_clean = pd.DataFrame(data_rows)
def plot_comparison(df, group_ids, title, y_var, hue_col='编号'):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    subset = df[df['编号'].isin(group_ids)]
    sns.lineplot(data=subset, x='温度', y=y_var, hue=hue_col, marker='o', linewidth=2.5)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('温度 (°C)', fontsize=12)
    plt.ylabel(y_var, fontsize=12)
    plt.legend(title='实验编号')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
group_1_2 = ['A12', 'B1']
plot_comparison(df_clean, group_1_2, '装料方式 I vs II 对 C4烯烃选择性的影响 (Co负载1wt%, 质量比1:1, 乙醇1.68ml/min)', 'C4烯烃选择性(%)')
plot_comparison(df_clean, group_1_2, '装料方式 I vs II 对乙醇转化率的影响 (Co负载1wt%, 质量比1:1, 乙醇1.68ml/min)', '乙醇转化率(%)')
a12_350 = df_clean[(df_clean['编号']=='A12') & (df_clean['温度']==400)]
b1_350 = df_clean[(df_clean['编号']=='B1') & (df_clean['温度']==400)]
print(f"--- 装料方式对照 (A12 vs B1) 在 400°C ---")
print(f"A12 (方式I): 转化率={a12_350['乙醇转化率(%)'].values[0]:.1f}%, C4选择性={a12_350['C4烯烃选择性(%)'].values[0]:.2f}%")
print(f"B1 (方式II): 转化率={b1_350['乙醇转化率(%)'].values[0]:.1f}%, C4选择性={b1_350['C4烯烃选择性(%)'].values[0]:.2f}%")
print(f"方式II 使 乙醇转化率 提高了 {b1_350['乙醇转化率(%)'].values[0] - a12_350['乙醇转化率(%)'].values[0]:.2f} 个百分点,使 C4选择性 提高了 {b1_350['C4烯烃选择性(%)'].values[0] - a12_350['C4烯烃选择性(%)'].values[0]:.2f} 个百分点")
features = [
    'Co负载量(wt%)',
    'Co/SiO2质量(mg)',
    'HAP质量(mg)',
    '乙醇浓度(ml/min)',
    '温度'
]
X = df_clean[features]
y_conversion = df_clean['乙醇转化率(%)']
y_selectivity = df_clean['C4烯烃选择性(%)']
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid = {'ridge__alpha': np.logspace(-3, 3, 50)}
grid_conv = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_conv.fit(X, y_conversion)

best_alpha_conv = grid_conv.best_params_['ridge__alpha']
coef_conv = grid_conv.best_estimator_['ridge'].coef_
intercept_conv = grid_conv.best_estimator_['ridge'].intercept_

print("========== 模型1：乙醇转化率 岭回归结果 ==========")
print(f"最优 alpha: {best_alpha_conv:.4f}")
print(f"截距: {intercept_conv:.4f}")
print("标准化回归系数:")
for feat, coef in zip(features, coef_conv):
    print(f"  {feat}: {coef:.4f}")

grid_sel = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_sel.fit(X, y_selectivity)
best_alpha_sel = grid_sel.best_params_['ridge__alpha']
coef_sel = grid_sel.best_estimator_['ridge'].coef_
intercept_sel = grid_sel.best_estimator_['ridge'].intercept_
print("========== 模型2：C4烯烃选择性 岭回归结果 ==========")
print(f"最优 alpha: {best_alpha_sel:.4f}")
print(f"截距: {intercept_sel:.4f}")
print("标准化回归系数:")
for feat, coef in zip(features, coef_sel):
    print(f"  {feat}: {coef:.4f}")
print("\n")
alphas_ = np.logspace(-3, 5, 100)
coefs = []

for a in alphas_:
    ridge = Ridge(alpha=a)
    pipeline_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', ridge)
    ])
    pipeline_temp.fit(X, y_conversion)
    coefs.append(pipeline_temp['ridge'].coef_)
plt.figure(figsize=(10, 7))
coefs = np.array(coefs)
for i, feat in enumerate(features):
    plt.plot(alphas_, coefs[:, i], label=feat)
plt.axvline(best_alpha_conv, color='red', linestyle='--', label=f'最优 alpha = {best_alpha_conv:.4f}')
plt.xscale('log')
plt.xlim(1e-3, 1e5)
plt.xlabel('Alpha')
plt.ylabel('标准化回归系数')
plt.title('岭回归系数路径')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()