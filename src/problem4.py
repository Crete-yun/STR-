import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from src.utils import PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, logger, RAW_DIR

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_and_preprocess_data():
    """加载原始数据和降噪后数据，并预处理"""
    logger.info("加载原始数据和降噪数据")
    
    # 读取数据
    raw_df = pd.read_excel(RAW_DIR / "attachment1.xlsx", engine='openpyxl')
    denoised_df = pd.read_excel(RAW_DIR / "attachment4.xlsx", engine='openpyxl')
    
    # 确保数据类型正确
    for df in [raw_df, denoised_df]:
        for i in range(1, 101):
            height_col = f'Height {i}'
            if height_col in df.columns:
                df[height_col] = pd.to_numeric(df[height_col], errors='coerce').fillna(0)
    
    # 合并数据集
    merged_data = []
    logger.info("处理样本数据...")
    
    # 获取所有唯一的样本-基因座组合
    sample_markers = raw_df[['Sample File', 'Marker']].drop_duplicates()
    
    for _, row in sample_markers.iterrows():
        sample = row['Sample File']
        marker = row['Marker']
        
        # 获取原始数据组
        raw_group = raw_df[(raw_df['Sample File'] == sample) & 
                          (raw_df['Marker'] == marker)]
        
        # 获取降噪数据组
        denoised_group = denoised_df[(denoised_df['Sample File'] == sample) & 
                                    (denoised_df['Marker'] == marker)]
        
        if raw_group.empty:
            continue
            
        # 收集该基因座的所有高度值
        marker_heights = []
        for i in range(1, 101):
            height_col = f'Height {i}'
            if height_col in raw_group.columns:
                height_val = raw_group[height_col].values[0]
                if not pd.isna(height_val) and height_val > 0:
                    marker_heights.append(height_val)
        
        total_height = sum(marker_heights)
        num_peaks = len(marker_heights)
        
        # 处理每个等位基因
        for i in range(1, 101):
            allele_col = f'Allele {i}'
            size_col = f'Size {i}'
            height_col = f'Height {i}'
            
            if allele_col not in raw_group.columns:
                continue
                
            raw_allele = raw_group[allele_col].values[0]
            raw_size = raw_group[size_col].values[0]
            raw_height = raw_group[height_col].values[0]
            
            # 检查是否为有效数据
            if pd.isna(raw_allele) or raw_height <= 0:
                continue
            
            # 在降噪数据中查找匹配
            denoised_height = 0
            if not denoised_group.empty:
                # 检查所有可能的等位基因列
                for j in range(1, 101):
                    d_allele_col = f'Allele {j}'
                    d_size_col = f'Size {j}'
                    d_height_col = f'Height {j}'
                    
                    if d_allele_col in denoised_group.columns:
                        d_allele = denoised_group[d_allele_col].values[0]
                        d_size = denoised_group[d_size_col].values[0]
                        d_height = denoised_group[d_height_col].values[0]
                        
                        if str(d_allele) == str(raw_allele) and abs(d_size - raw_size) < 0.1:
                            denoised_height = d_height
                            break
            
            # 计算高度比例
            height_ratio = raw_height / total_height if total_height > 0 else 0
            
            merged_data.append({
                'sample': sample,
                'marker': marker,
                'allele': raw_allele,
                'size': raw_size,
                'raw_height': raw_height,
                'denoised_height': denoised_height,
                'height_ratio': height_ratio,
                'total_height': total_height,
                'num_peaks': num_peaks,
                'is_ol': 1 if 'OL' in str(raw_allele) else 0
            })
    
    logger.info(f"预处理完成，共处理 {len(merged_data)} 条记录")
    return pd.DataFrame(merged_data)

def convert_to_long_format(df):
    """将宽格式数据转换为长格式"""
    long_data = []
    
    for _, row in df.iterrows():
        sample = row['Sample File']
        marker = row['Marker']
        
        for i in range(1, 101):
            allele_col = f'Allele {i}'
            size_col = f'Size {i}'
            height_col = f'Height {i}'
            
            if allele_col in row and not pd.isna(row[allele_col]) and row[allele_col] != '':
                long_data.append({
                    'Sample File': sample,
                    'Marker': marker,
                    'Allele': row[allele_col],
                    'Size': row[size_col] if size_col in row else np.nan,
                    'Height': row[height_col] if height_col in row else np.nan
                })
    
    return pd.DataFrame(long_data)

def visualize_denoising_results(raw_df, denoised_df, results_dir):
    """
    生成降噪结果可视化图表并保存到results目录
    包括四种图表：
    1. 降噪前后对比图 (problem4_denoise_comparison.png)
    2. 相关性分布图 (problem4_correlation_distribution.png)
    3. 降噪示例图 (problem4_denoise_example.png)
    4. 噪声减少分布图 (problem4_noise_reduction_distribution.png)
    """
    # 确保结果目录存在
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 将数据转换为长格式
    raw_long = convert_to_long_format(raw_df)
    denoised_long = convert_to_long_format(denoised_df)
    
    # 合并原始和降噪数据
    combined = pd.merge(
        raw_long, 
        denoised_long, 
        on=['Sample File', 'Marker', 'Allele', 'Size'], 
        suffixes=('_raw', '_denoised')
    )
    
    # 过滤有效数据点
    combined = combined.dropna(subset=['Height_raw', 'Height_denoised'])
    combined = combined[(combined['Height_raw'] > 0) & (combined['Height_denoised'] >= 0)]
    
    # 如果没有足够数据，跳过可视化
    if combined.empty:
        logger.warning("没有足够数据生成可视化图表")
        return
    
    # 示例1: 降噪前后对比图
    plt.figure(figsize=(12, 6))
    
    # 选择一个示例样本和标记
    sample_marker = combined[['Sample File', 'Marker']].drop_duplicates().iloc[0]
    sample_name = sample_marker['Sample File']
    marker_name = sample_marker['Marker']
    
    sample_data = combined[
        (combined['Sample File'] == sample_name) & 
        (combined['Marker'] == marker_name)
    ]
    
    # 原始数据
    plt.subplot(1, 2, 1)
    plt.bar(sample_data['Size'], sample_data['Height_raw'], width=0.5, alpha=0.7, color='blue')
    plt.xlabel('等位基因大小')
    plt.ylabel('峰高')
    plt.title(f'原始数据\n{sample_name} - {marker_name}')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 降噪后数据
    plt.subplot(1, 2, 2)
    plt.bar(sample_data['Size'], sample_data['Height_denoised'], width=0.5, alpha=0.7, color='red')
    plt.xlabel('等位基因大小')
    plt.ylabel('峰高')
    plt.title(f'降噪后数据\n{sample_name} - {marker_name}')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'problem4_denoise_comparison.png', dpi=300)
    plt.close()
    
    # 示例2: 相关性分布图
    plt.figure(figsize=(8, 6))
    correlation = np.corrcoef(combined['Height_raw'], combined['Height_denoised'])[0, 1]
    sns.regplot(x=combined['Height_raw'], y=combined['Height_denoised'], 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.xlabel('原始峰高')
    plt.ylabel('降噪后峰高')
    plt.title(f'峰高相关性分布 (r = {correlation:.2f})')
    plt.grid(True)
    plt.savefig(results_dir / 'problem4_correlation_distribution.png', dpi=300)
    plt.close()
    
    # 示例3: 降噪示例图
    plt.figure(figsize=(8, 6))
    
    # 选择AMEL标记的样本
    amel_sample = combined[combined['Marker'] == 'AMEL'].iloc[0]
    amel_data = combined[
        (combined['Sample File'] == amel_sample['Sample File']) & 
        (combined['Marker'] == 'AMEL')
    ]
    
    plt.bar(amel_data['Size'], amel_data['Height_raw'], width=0.5, alpha=0.7, label='原始数据')
    plt.bar(amel_data['Size'], amel_data['Height_denoised'], width=0.3, alpha=0.7, label='降噪后数据')
    plt.xlabel('等位基因大小')
    plt.ylabel('峰高')
    plt.title(f'降噪示例\n{amel_sample["Sample File"]} - AMEL')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'problem4_denoise_example.png', dpi=300)
    plt.close()
    
    # 示例4: 噪声减少分布图
    combined['noise_reduction'] = combined['Height_raw'] - combined['Height_denoised']
    
    plt.figure(figsize=(8, 6))
    plt.hist(combined['noise_reduction'], bins=50, alpha=0.7)
    plt.xlabel('噪声减少量')
    plt.ylabel('频率')
    plt.title('噪声减少量分布')
    plt.grid(True)
    plt.savefig(results_dir / 'problem4_noise_reduction_distribution.png', dpi=300)
    plt.close()
    
    logger.info(f"生成4个可视化图表并保存到 {results_dir}")

def train_denoising_model():
    """训练降噪模型"""
    logger.info("开始训练降噪模型")
    
    # 加载并预处理数据
    df = load_and_preprocess_data()
    
    # 创建标签 (0=噪声, 1=信号)
    df['label'] = (df['denoised_height'] > 0).astype(int)
    
    # 准备特征和标签
    features = df[['raw_height', 'height_ratio', 'total_height', 'num_peaks', 'is_ol']]
    labels = df['label']
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    logger.info(f"正例比例: {labels.mean():.2f}")
    
    # 创建并训练模型
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    )
    
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info("降噪模型评估报告:\n" + report)
    
    # 保存模型
    model_path = MODEL_DIR / "denoising_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"降噪模型已保存至 {model_path}")
    
    return model

def apply_denoising(input_file, output_file, model):
    """应用降噪模型到新数据"""
    logger.info(f"对 {input_file} 应用降噪模型")
    
    # 读取数据
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # 确保数据类型正确
    for i in range(1, 101):
        height_col = f'Height {i}'
        if height_col in df.columns:
            df[height_col] = pd.to_numeric(df[height_col], errors='coerce').fillna(0)
    
    # 保存原始数据用于比较
    original_df = df.copy()
    
    # 处理每个样本的每个基因座
    for (sample, marker), group in df.groupby(['Sample File', 'Marker']):
        # 收集该基因座的所有高度值
        marker_heights = []
        for i in range(1, 101):
            height_col = f'Height {i}'
            if height_col in group.columns:
                height_val = group[height_col].values[0]
                if not pd.isna(height_val) and height_val > 0:
                    marker_heights.append(height_val)
        
        total_height = sum(marker_heights)
        num_peaks = len(marker_heights)
        
        # 处理每个等位基因
        for i in range(1, 101):
            allele_col = f'Allele {i}'
            height_col = f'Height {i}'
            
            if allele_col not in group.columns:
                continue
                
            allele_val = group[allele_col].values[0]
            height_val = group[height_col].values[0]
            
            # 跳过无效数据
            if pd.isna(allele_val) or height_val <= 0:
                continue
            
            # 计算特征
            height_ratio = height_val / total_height if total_height > 0 else 0
            
            features = pd.DataFrame([{
                'raw_height': height_val,
                'height_ratio': height_ratio,
                'total_height': total_height,
                'num_peaks': num_peaks,
                'is_ol': 1 if 'OL' in str(allele_val) else 0
            }])
            
            # 预测并更新高度
            prediction = model.predict(features)[0]
            if prediction == 0:  # 噪声
                df.loc[group.index, height_col] = 0
    
    # 保存降噪后的数据
    df.to_excel(output_file, index=False)
    logger.info(f"降噪结果已保存至 {output_file}")
    
    # 生成可视化结果
    visualize_denoising_results(original_df, df, RESULTS_DIR)
    
    return df

def solve_problem4():
    """解决问题4：噪声抑制"""
    # 训练或加载模型
    model_path = MODEL_DIR / "denoising_model.pkl"
    if model_path.exists():
        logger.info("加载现有降噪模型")
        model = joblib.load(model_path)
    else:
        logger.info("训练新降噪模型")
        model = train_denoising_model()
    
    # 应用降噪
    input_file = RAW_DIR / "attachment4.xlsx"
    output_file = PROCESSED_DIR / "denoised_attachment4.xlsx"
    
    # 确保输出目录存在
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    apply_denoising(input_file, output_file, model)
    logger.info("噪声抑制处理完成")

if __name__ == "__main__":
    solve_problem4()