# problem2.py (修复优化问题)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.utils import load_processed_data, load_metadata, load_genotype_data, RESULTS_DIR
import time
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def build_genotype_dict(genotype_db):
    """构建基因型字典（添加空值处理）"""
    genotype_dict = {}
    for _, row in genotype_db.iterrows():
        sample_id = str(row['Sample ID'])
        marker = row['Marker']
        allele = row['Allele']
        
        if pd.isna(sample_id) or pd.isna(marker) or pd.isna(allele):
            continue
            
        if sample_id not in genotype_dict:
            genotype_dict[sample_id] = {}
        if marker not in genotype_dict[sample_id]:
            genotype_dict[sample_id][marker] = set()
        
        genotype_dict[sample_id][marker].add(str(allele))
    return genotype_dict

def build_contribution_matrix(group, genotype_dict, contributors):
    """构建贡献矩阵（添加空值处理）"""
    alleles = group['Allele'].values
    markers = group['Marker'].values
    n_alleles = len(alleles)
    n_contributors = len(contributors)
    
    contribution_matrix = np.zeros((n_alleles, n_contributors))
    
    for i, (allele, marker) in enumerate(zip(alleles, markers)):
        for j, contributor in enumerate(contributors):
            contrib_id = str(contributor)
            if contrib_id in genotype_dict and marker in genotype_dict[contrib_id]:
                if str(allele) in genotype_dict[contrib_id][marker]:
                    contribution_matrix[i, j] = 1
    return contribution_matrix

def estimate_initial_ratio(contribution_matrix, heights):
    """基于贡献矩阵和峰高估计初始比例"""
    n_contributors = contribution_matrix.shape[1]
    
    # 计算每个贡献者的总峰高
    contributor_heights = np.zeros(n_contributors)
    for j in range(n_contributors):
        idx = np.where(contribution_matrix[:, j] > 0)[0]
        if len(idx) > 0:
            contributor_heights[j] = np.sum(heights[idx])
    
    # 移除全零列（无贡献的贡献者）
    valid_contributors = contributor_heights > 0
    if np.sum(valid_contributors) == 0:
        return None
    
    # 归一化
    total = np.sum(contributor_heights[valid_contributors])
    if total > 0:
        initial_ratio = np.zeros(n_contributors)
        initial_ratio[valid_contributors] = contributor_heights[valid_contributors] / total
        return initial_ratio
    else:
        return np.ones(n_contributors) / n_contributors

def solve_problem2():
    print("="*50)
    print("问题2：贡献者比例估计")
    print("="*50)
    start_time = time.time()
    
    # 加载数据
    data = load_processed_data(2)
    metadata = load_metadata()
    genotype_db = load_genotype_data()
    
    if data is None or not metadata or genotype_db is None:
        print("错误：未找到必要数据")
        return
    
    # 构建基因型字典
    genotype_dict = build_genotype_dict(genotype_db)
    
    results = []
    mae_list = []
    processed_samples = 0
    skipped_samples = 0
    total_samples = len(data.groupby('Sample File'))
    
    for sample_file, group in data.groupby('Sample File'):
        processed_samples += 1
        print(f"处理样本 {processed_samples}/{total_samples}: {sample_file}")
        
        sample_meta = metadata.get(sample_file, {})
        true_ratio = sample_meta.get('ratio')
        contributors = sample_meta.get('contributors')
        
        if not true_ratio or not contributors:
            print(f"警告: 样本{sample_file}缺少元数据，跳过")
            skipped_samples += 1
            continue
        
        n_contributors = len(true_ratio)
        true_ratio = np.array(true_ratio)
        
        # 准备数据
        observed_heights = group['Height'].values
        
        # 构建贡献矩阵
        contribution_matrix = build_contribution_matrix(
            group, genotype_dict, contributors
        )
        
        # 检查贡献矩阵是否有效
        if np.sum(contribution_matrix) == 0:
            print(f"警告: 样本{sample_file}贡献矩阵无效，跳过")
            skipped_samples += 1
            continue
        
        # 估计初始比例
        x0 = estimate_initial_ratio(contribution_matrix, observed_heights)
        if x0 is None:
            print(f"警告: 样本{sample_file}初始比例估计失败，跳过")
            skipped_samples += 1
            continue
        
        # 优化比例
        def loss(p):
            expected = contribution_matrix @ p
            # 添加L2正则化防止过拟合
            reg_lambda = 0.05
            reg_term = reg_lambda * np.sum(p**2)
            return np.sum((np.sqrt(observed_heights) - np.sqrt(expected))**2) + reg_term
        
        # 约束条件
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: np.min(x) - 0.01}  # 最小比例1%
        )
        
        bounds = [(0.01, 0.99) for _ in range(n_contributors)]
        
        # 优化
        res = minimize(loss, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        
        if res.success:
            pred_ratio = res.x
            opt_success = True
            opt_message = "成功"
        else:
            print(f"优化失败: {sample_file} - {res.message}")
            pred_ratio = x0
            opt_success = False
            opt_message = res.message
        
        # 计算MAE
        mae = np.mean(np.abs(true_ratio - pred_ratio))
        mae_list.append(mae)
        
        results.append({
            'sample': sample_file,
            'true_ratio': true_ratio.tolist(),
            'pred_ratio': pred_ratio.tolist(),
            'optimization_success': opt_success,
            'optimization_message': opt_message,
            'mae': float(mae)
        })
    
    # 计算整体评估指标
    if results:
        mae = np.mean(mae_list)
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"处理耗时: {time.time()-start_time:.2f}秒")
        print(f"成功处理样本: {len(results)}")
        print(f"跳过样本: {skipped_samples}")
        
        # 保存结果
        with open(RESULTS_DIR / "problem2_results.json", 'w') as f:
            json.dump({
                'results': results,
                'overall_mae': float(mae),
                'num_samples': int(len(results)),
                'skipped_samples': skipped_samples
            }, f, indent=4)
        
        # 可视化结果
        if len(results) > 0:
            plt.figure(figsize=(12, 8))
            samples_to_show = min(30, len(results))
            for i, result in enumerate(results[:samples_to_show]):
                plt.subplot(5, 6, i+1)
                n_contributors = len(result['true_ratio'])
                x = np.arange(n_contributors)
                width = 0.35
                plt.bar(x - width/2, result['true_ratio'], width, alpha=0.7, label='真实值')
                plt.bar(x + width/2, result['pred_ratio'], width, alpha=0.7, label='预测值')
                plt.title(f"{result['sample'][:10]}...")
                plt.ylim(0, 1)
                if i == 0:
                    plt.legend()
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'problem2_ratio_comparison.png', dpi=300)
            plt.close()
            
            # 绘制MAE分布图
            plt.figure(figsize=(10, 6))
            plt.hist(mae_list, bins=20)
            plt.xlabel('平均绝对误差 (MAE)')
            plt.ylabel('样本数量')
            plt.title('比例估计误差分布')
            plt.savefig(RESULTS_DIR / 'problem2_mae_distribution.png', dpi=300)
            plt.close()
        
        print(f"\n处理完成！结果已保存")
    else:
        print("未找到有效样本进行比例估计")

if __name__ == "__main__":
    solve_problem2()