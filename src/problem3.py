# problem3.py (改进匹配效率)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import load_processed_data, load_metadata, load_genotype_data, RESULTS_DIR
from collections import defaultdict
import random
import itertools
from tqdm import tqdm
# 在文件开头添加以下代码
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
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

def calculate_combo_score(candidates, observed_alleles, genotype_dict):
    """计算候选组合的匹配分数（改进版）"""
    score = 0
    for marker, alleles in observed_alleles.items():
        found = set()
        for candidate in candidates:
            if candidate in genotype_dict and marker in genotype_dict[candidate]:
                found |= genotype_dict[candidate][marker]
        
        matched = found & alleles
        if matched == alleles:
            score += 2
        elif len(matched) > 0:
            score += 1
        else:
            score -= 0.5  # 惩罚不匹配的标记
    
    return score

def find_contributors(observed_alleles, genotype_dict, n_contributors, top_n=30, max_iter=5000):
    """寻找最可能的贡献者（改进算法）"""
    # 构建候选者分数字典
    candidate_scores = defaultdict(int)
    for candidate_id, candidate_genotype in genotype_dict.items():
        for marker, alleles in observed_alleles.items():
            if marker in candidate_genotype:
                match_count = len(candidate_genotype[marker] & alleles)
                if match_count == len(candidate_genotype[marker]):
                    candidate_scores[candidate_id] += 2
                elif match_count > 0:
                    candidate_scores[candidate_id] += 1
    
    # 按分数排序
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [c[0] for c in sorted_candidates[:top_n]]
    
    # 如果没有足够候选者，补充随机候选者
    if len(top_candidates) < n_contributors:
        all_candidates = list(genotype_dict.keys())
        additional_candidates = random.sample(
            [c for c in all_candidates if c not in top_candidates], 
            min(len(all_candidates) - len(top_candidates), n_contributors - len(top_candidates))
        )
        top_candidates.extend(additional_candidates)
    
    # 如果候选者数量不多，使用精确搜索
    if len(top_candidates) <= 10 or n_contributors == 1:
        best_match = []
        best_score = -np.inf
        
        for combo in itertools.combinations(top_candidates, n_contributors):
            combo_score = calculate_combo_score(combo, observed_alleles, genotype_dict)
            if combo_score > best_score:
                best_score = combo_score
                best_match = combo
    else:
        # 否则使用随机搜索
        best_match = []
        best_score = -np.inf
        
        for _ in range(max_iter):
            combo = random.sample(top_candidates, n_contributors)
            combo_score = calculate_combo_score(combo, observed_alleles, genotype_dict)
            
            if combo_score > best_score:
                best_score = combo_score
                best_match = combo
    
    return list(best_match), best_score

def solve_problem3():
    print("="*50)
    print("问题3：贡献者基因型推断")
    print("="*50)
    
    # 加载数据
    data1 = load_processed_data(1)
    data2 = load_processed_data(2)
    if data1 is None or data2 is None:
        print("错误：无法加载附件1或附件2数据")
        return
    
    combined_data = pd.concat([data1, data2])
    metadata = load_metadata()
    genotype_db = load_genotype_data()
    
    if combined_data.empty or not metadata or genotype_db is None:
        print("错误：未找到必要数据")
        return
    
    genotype_dict = build_genotype_dict(genotype_db)
    
    results = []
    processed_samples = 0
    skipped_samples = 0
    total_samples = len(combined_data.groupby('Sample File'))
    
    for sample_file, group in tqdm(combined_data.groupby('Sample File'), total=total_samples):
        processed_samples += 1
        print(f"\n处理样本 {processed_samples}/{total_samples}: {sample_file}")
        
        sample_meta = metadata.get(sample_file)
        if not sample_meta:
            print(f"警告: 样本{sample_file}缺少元数据，跳过")
            skipped_samples += 1
            continue
        
        true_contributors = sample_meta['contributors']
        n_contributors = len(true_contributors)
        
        # 获取样本的等位基因集合
        observed_alleles = {}
        for marker, marker_group in group.groupby('Marker'):
            observed_alleles[marker] = set(marker_group['Allele'].astype(str))
        
        # 使用改进算法寻找贡献者
        best_match, score = find_contributors(
            observed_alleles, 
            genotype_dict, 
            n_contributors
        )
        
        # 评估匹配结果 (顺序无关)
        true_contributors_str = [str(c) for c in true_contributors]
        true_positives = len(set(best_match) & set(true_contributors_str))
        accuracy = true_positives / n_contributors if n_contributors > 0 else 0
        
        results.append({
            'sample': sample_file,
            'true_contributors': true_contributors,
            'predicted_contributors': list(best_match),
            'accuracy': accuracy,
            'matching_score': score,
            'n_contributors': n_contributors
        })
        
        print(f"真实贡献者: {true_contributors}")
        print(f"预测贡献者: {best_match}")
        print(f"准确率: {accuracy:.2f} | 匹配分数: {score}")
    
    if results:
        overall_accuracy = np.mean([r['accuracy'] for r in results])
        print(f"整体准确率: {overall_accuracy:.4f}")
        print(f"成功处理样本: {len(results)}")
        print(f"跳过样本: {skipped_samples}")
        
        with open(RESULTS_DIR / "problem3_results.json", 'w') as f:
            json.dump({
                'results': results,
                'overall_accuracy': overall_accuracy,
                'num_samples': len(results),
                'skipped_samples': skipped_samples
            }, f, indent=4)
        
        # 绘制准确率分布图
        plt.figure(figsize=(10, 6))
        plt.hist([r['accuracy'] for r in results], bins=20)
        plt.xlabel('准确率')
        plt.ylabel('频率')
        plt.title('准确率分布图')
        plt.savefig(RESULTS_DIR / 'problem3_accuracy_distribution.png', dpi=300)
        plt.close()
        
        print(f"\n处理完成！结果已保存")
    else:
        print("未找到有效样本进行基因型推断")

if __name__ == "__main__":
    solve_problem3()