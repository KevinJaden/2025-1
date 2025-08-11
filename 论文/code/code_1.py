import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 坐标轴负号的处理
plt.rcParams['axes.unicode_minus'] = False
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

data_1 = pd.read_excel('附件1.xlsx')
data_2 = pd.read_excel('附件2.xlsx')
data_1['催化剂组合编号'] = data_1['催化剂组合编号'].ffill()
data_1['催化剂组合'] = data_1['催化剂组合'].ffill()
data_needed = data_1[['催化剂组合编号', '温度', '乙醇转化率(%)', 'C4烯烃选择性(%)']].copy()
data_needed['温度'] = data_needed['温度'].astype(int)
data_needed['乙醇转化率(%)'] = pd.to_numeric(data_needed['乙醇转化率(%)'], errors='coerce')
data_needed['C4烯烃选择性(%)'] = pd.to_numeric(data_needed['C4烯烃选择性(%)'], errors='coerce')

def interpolate_large_gaps(group):
    sorted_group = group.sort_values('温度').reset_index(drop=True)
    temps = sorted_group['温度'].values
    conv = sorted_group['乙醇转化率(%)'].values
    c4sel = sorted_group['C4烯烃选择性(%)'].values
    result_rows = []
    for i in range(len(temps) - 1):
        t0, t1 = temps[i], temps[i+1]
        gap = t1 - t0
        if gap > 25:
            print(f"[{sorted_group['催化剂组合编号'].iloc[0]}] 在 {t0} 和 {t1} 之间发现温差 {gap}，需插值")
            if len(temps) >= 4:
                method = 'cubic'
            elif len(temps) >= 2:
                method = 'linear'
            else:
                method = 'linear'
            try:
                f_conv = interp1d([t0, t1], [conv[i], conv[i+1]], kind=method, bounds_error=False)
                f_c4sel = interp1d([t0, t1], [c4sel[i], c4sel[i+1]], kind=method, bounds_error=False)
            except:
                f_conv = lambda x: np.interp(x, [t0, t1], [conv[i], conv[i+1]])
                f_c4sel = lambda x: np.interp(x, [t0, t1], [c4sel[i], c4sel[i+1]])
            insert_temps = np.arange(t0 + 25, t1, 25)
            result_rows.append({
                '催化剂组合编号': sorted_group['催化剂组合编号'].iloc[0],
                '温度': t0,
                '乙醇转化率(%)': conv[i],
                'C4烯烃选择性(%)': c4sel[i]
            })
            for temp in insert_temps:
                result_rows.append({
                    '催化剂组合编号': sorted_group['催化剂组合编号'].iloc[0],
                    '温度': temp,
                    '乙醇转化率(%)': float(f_conv(temp)),
                    'C4烯烃选择性(%)': float(f_c4sel(temp))
                })
        else:
            result_rows.append({
                '催化剂组合编号': sorted_group['催化剂组合编号'].iloc[0],
                '温度': t0,
                '乙醇转化率(%)': conv[i],
                'C4烯烃选择性(%)': c4sel[i]
            })
    result_rows.append({
        '催化剂组合编号': sorted_group['催化剂组合编号'].iloc[0],
        '温度': temps[-1],
        '乙醇转化率(%)': conv[-1],
        'C4烯烃选择性(%)': c4sel[-1]
    })
    return pd.DataFrame(result_rows)
final_list = []
for name, group in data_needed.groupby('催化剂组合编号'):
    processed_group = interpolate_large_gaps(group)
    final_list.append(processed_group)
final_result = pd.concat(final_list, ignore_index=True)
final_result = final_result.sort_values(['催化剂组合编号', '温度']).reset_index(drop=True)
print("插值完成")
final_result.to_excel('附件1改.xlsx', index=False)
catalysts = final_result['催化剂组合编号'].unique()
n = len(catalysts)
cols = 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()
colors = sns.color_palette("husl", 2)
for idx, cat in enumerate(catalysts):
    ax1 = axes[idx]
    subset = final_result[final_result['催化剂组合编号'] == cat].sort_values('温度')
    ax1.plot(subset['温度'], subset['乙醇转化率(%)'], marker='o', label='乙醇转化率 (%)', color=colors[0])
    ax1.set_xlabel('温度 (°C)')
    ax1.set_ylabel('乙醇转化率 (%)', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(subset['温度'], subset['C4烯烃选择性(%)'], marker='s', label='C4烯烃选择性 (%)', color=colors[1])
    ax2.set_ylabel('C4烯烃选择性 (%)', color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    ax1.set_title(f'催化剂组合: {cat}')
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig('论文/图/1-1-1.png', dpi=300, bbox_inches='tight')
plt.show()
colors = sns.color_palette("husl", 2)
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(data_2['时间（min）'], data_2['乙醇转化率(%)'], 'o-', color=colors[0],
         label='乙醇转化率 (%)', linewidth=2.5, markersize=6)
ax1.set_xlabel('反应时间 (min)', fontsize=12)
ax1.set_ylabel('乙醇转化率 (%)', color=colors[0], fontsize=12)
ax1.tick_params(axis='y', labelcolor=colors[0], length=6, width=1.2)
ax1.tick_params(axis='x', length=6, width=1.2)
ax1.grid(True, linestyle='--', alpha=0.5)
ax2 = ax1.twinx()
ax2.plot(data_2['时间（min）'], data_2['C4烯烃选择性'], 's-', color=colors[1],
         label='C4烯烃选择性 (%)', linewidth=2.5, markersize=6)
ax2.set_ylabel('C4烯烃选择性 (%)', color=colors[1], fontsize=12)
ax2.tick_params(axis='y', labelcolor=colors[1], length=6, width=1.2)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.3, 0.2), fontsize=11)
fig.tight_layout()
plt.savefig('论文/图/1-2-1.png', dpi=300, bbox_inches='tight')
plt.show()
catalysts = sorted(data_1['催化剂组合编号'].unique(), key=lambda x: (x[0], int(x[1:])))
colors = sns.color_palette("husl", len(catalysts))
correlation_results = []
for cat in catalysts:
    subset = data_1[data_1['催化剂组合编号'] == cat][['温度', '乙醇转化率(%)', 'C4烯烃选择性(%)']].dropna()
    if len(subset) < 2:
        corr_conv_temp = np.nan
        corr_sel_temp = np.nan
    else:
        corr_conv_temp = subset['乙醇转化率(%)'].corr(subset['温度'])
        corr_sel_temp = subset['C4烯烃选择性(%)'].corr(subset['温度'])
    
    correlation_results.append({
        '催化剂组合': cat,
        '乙醇转化率-温度 (r)': corr_conv_temp,
        'C4烯烃选择性-温度 (r)': corr_sel_temp
    })
corr_df = pd.DataFrame(correlation_results)
print("\n=== 各催化剂组合：乙醇转化率、C4烯烃选择性 与 温度 的皮尔逊相关系数 ===")
print(corr_df.round(4).to_string(index=False))
plt.figure(figsize=(10, 8))
size = 100 + 300 * (corr_df['乙醇转化率-温度 (r)'] + corr_df['C4烯烃选择性-温度 (r)'])
scatter = plt.scatter(
    corr_df['乙醇转化率-温度 (r)'], 
    corr_df['C4烯烃选择性-温度 (r)'],
    s=size, c=size, cmap='viridis', alpha=0.8, edgecolors='k', linewidth=0.5
)
for i, txt in enumerate(corr_df['催化剂组合']):
    plt.annotate(txt, (corr_df['乙醇转化率-温度 (r)'][i], corr_df['C4烯烃选择性-温度 (r)'][i]),
                 fontsize=9, ha='center', va='center')
plt.axhline(0.95, color='r', linestyle='--', alpha=0.5, label='选择性高相关阈值 (0.95)')
plt.axvline(0.95, color='b', linestyle='--', alpha=0.5, label='转化率高相关阈值 (0.95)')
plt.xlabel('乙醇转化率与温度的皮尔逊相关系数')
plt.ylabel('C4烯烃选择性与温度的皮尔逊相关系数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('论文/图/1-2-2.png', dpi=300, bbox_inches='tight')
plt.show()