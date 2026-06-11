import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df1 = pd.read_excel(r'C:\Users\86187\OneDrive\文档\桌面\1.xlsx')  # 原始数据
df2 = pd.read_excel(r'C:\Users\86187\OneDrive\文档\桌面\2.xlsx')  # 降噪后数据

# 准备数据 - 提取几个关键标记物的高度信息进行比较
markers = ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358']

# 创建对比图
plt.figure(figsize=(15, 10))

for i, marker in enumerate(markers, 1):
    # 原始数据
    orig_row = df1[df1['Marker'] == marker].iloc[0]
    orig_heights = [h for h in orig_row[5::3] if isinstance(h, (int, float)) and h > 0]
    
    # 降噪后数据
    denoised_row = df2[df2['Marker'] == marker].iloc[0]
    denoised_heights = [h for h in denoised_row[5::3] if isinstance(h, (int, float)) and h > 0]
    
    # 绘制子图
    plt.subplot(2, 3, i)
    plt.bar(np.arange(len(orig_heights))-0.2, orig_heights, width=0.4, label='Original', alpha=0.7)
    plt.bar(np.arange(len(denoised_heights))+0.2, denoised_heights, width=0.4, label='Denoised', alpha=0.7)
    plt.title(marker)
    plt.ylabel('Height')
    plt.xticks(np.arange(max(len(orig_heights), len(denoised_heights))))
    if i == 1:
        plt.legend()

plt.suptitle('降噪前后对比图')
plt.tight_layout()
plt.savefig('height_comparison.png')
plt.show()

# 创建指标对比表
comparison_table = pd.DataFrame(columns=['Marker', 'Original Peaks', 'Denoised Peaks', 
                                        'Original Max Height', 'Denoised Max Height',
                                        'Original Total Height', 'Denoised Total Height'])

for marker in markers:
    # 原始数据
    orig_row = df1[df1['Marker'] == marker].iloc[0]
    orig_heights = [h for h in orig_row[5::3] if isinstance(h, (int, float)) and h > 0]
    
    # 降噪后数据
    denoised_row = df2[df2['Marker'] == marker].iloc[0]
    denoised_heights = [h for h in denoised_row[5::3] if isinstance(h, (int, float)) and h > 0]
    
    comparison_table = comparison_table.append({
        'Marker': marker,
        'Original Peaks': len(orig_heights),
        'Denoised Peaks': len(denoised_heights),
        'Original Max Height': max(orig_heights) if orig_heights else 0,
        'Denoised Max Height': max(denoised_heights) if denoised_heights else 0,
        'Original Total Height': sum(orig_heights),
        'Denoised Total Height': sum(denoised_heights)
    }, ignore_index=True)

# 保存对比表
comparison_table.to_excel(r'C:\Users\86187\OneDrive\文档\桌面\comparison_table.xlsx', index=False)

print("对比图和对比表已生成完成!")