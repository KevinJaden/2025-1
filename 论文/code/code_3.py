import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 坐标轴负号的处理
plt.rcParams['axes.unicode_minus'] = False
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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
        mass_ratio = f"{co_sio2_mass}:{hap_mass}" if hap_mass != 0 else f"{co_sio2_mass}:0"
        hap_mass_fraction = hap_mass / total_mass if total_mass > 0 else 0
        current_info = {
            '催化剂组合编号': current_id,
            'Co负载量(wt%)': co_loading,
            'Co/SiO2质量(mg)': co_sio2_mass,
            'HAP质量(mg)': hap_mass,
            '总质量(mg)': total_mass,
            '装料比(Co/SiO2:HAP)': mass_ratio,
            'HAP质量分数': hap_mass_fraction,
            '乙醇浓度(ml/min)': ethanol_conc,
            '装料方式': 'I' if str(current_id).startswith('A') else 'II',
            '有无HAP': has_hap
        }
    if pd.notna(row.get('温度')) and np.isfinite(row['温度']):
        record = current_info.copy()
        record['温度'] = row['温度']
        record['乙醇转化率(%)'] = row['乙醇转化率(%)']
        record['C4烯烃选择性(%)'] = row['C4烯烃选择性(%)']
        if pd.notna(record['乙醇转化率(%)']) and pd.notna(record['C4烯烃选择性(%)']):
            data_rows.append(record)
df_clean = pd.DataFrame(data_rows)
df_clean['C4烯烃收率(%)'] = df_clean['乙醇转化率(%)'] * df_clean['C4烯烃选择性(%)'] / 100
print("\n【建立多项式回归模型】")
features = ['Co负载量(wt%)', 'HAP质量分数', '乙醇浓度(ml/min)', '温度']
X = df_clean[features]
y = df_clean['C4烯烃收率(%)']
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"模型R²得分: {r2:.4f}")

# 定义搜索的参数范围
co_loading_range = np.linspace(0.5, 5, 10)
hap_fraction_range = np.linspace(0, 1, 11)
ethanol_conc_range = np.linspace(0.3, 2.1, 7)
temperature_range = np.linspace(250, 450, 11)

# 创建网格
# 1. 寻找全局理论最优解
param_grid = np.array(np.meshgrid(co_loading_range, hap_fraction_range, ethanol_conc_range, temperature_range))
param_combinations = param_grid.reshape(4, -1).T
print(f"正在搜索 {param_combinations.shape[0]} 个参数组合...")
predicted_yields = model.predict(param_combinations)
best_idx = np.argmax(predicted_yields)
best_yield = predicted_yields[best_idx]
best_params = param_combinations[best_idx]
best_co, best_hap_frac, best_ethanol, best_temp = best_params
print(f"\n理论上的全局最优条件为：")
print(f"  - Co负载量: {best_co:.2f} wt%")
print(f"  - HAP质量分数: {best_hap_frac:.3f}")
print(f"  - 乙醇浓度: {best_ethanol:.2f} ml/min")
print(f"  - 温度: {best_temp:.1f} °C")
print(f"  - 预测的C4烯烃收率: {best_yield:.2f}%")

# 2. 寻找低温理论最优解
param_grid_low = np.array(np.meshgrid(co_loading_range, hap_fraction_range, ethanol_conc_range, temperature_range_low))
param_combinations_low = param_grid_low.reshape(4, -1).T
print(f"\n正在低温(350°C以下)范围内搜索 {param_combinations_low.shape[0]} 个参数组合...")
X_pred_low = pd.DataFrame(param_combinations_low, columns=features)
predicted_yields_low = model.predict(X_pred_low)
best_idx_low = np.argmax(predicted_yields_low)
best_yield_low = predicted_yields_low[best_idx_low]
best_params_low = param_combinations_low[best_idx_low]
best_co_low, best_hap_frac_low, best_ethanol_low, best_temp_low = best_params_low
print(f"\n理论上的低温最优条件为：")
print(f"  - Co负载量: {best_co_low:.2f} wt%")
print(f"  - HAP质量分数: {best_hap_frac_low:.3f}")
print(f"  - 乙醇浓度: {best_ethanol_low:.2f} ml/min")
print(f"  - 温度: {best_temp_low:.1f} °C")
print(f"  - 预测的C4烯烃收率: {best_yield_low:.2f}%")
print("\n理论最优解分析完成。")