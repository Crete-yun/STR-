# problem1.py (修复数据加载问题)
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.utils.class_weight import compute_class_weight
from src.utils import PROCESSED_DIR, RESULTS_DIR, load_metadata, load_processed_data

# 添加缺失的FEATURES_DIR定义
FEATURES_DIR = Path(__file__).resolve().parent.parent / "data" / "features"

def extract_features(df, metadata):
    """提取特征用于人数识别"""
    sample_features = []
    
    for sample_file, group in df.groupby('Sample File'):
        noc = metadata.get(sample_file, {}).get('contributor_count', 1)
        
        features = {
            'Sample File': sample_file,
            'TrueNOC': noc,
            'TotalMarkers': group['Marker'].nunique(),
            'TotalPeaks': len(group),
            'MaxPeaksPerMarker': group['Marker'].value_counts().max(),
            'MeanHeight': group['Height'].mean(),
            'MedianHeight': group['Height'].median(),
            'HeightCV': group['Height'].std() / (group['Height'].mean() + 1e-5),
            'HeightRange': group['Height'].max() - group['Height'].min(),
            'Height90Percentile': group['Height'].quantile(0.9),
            'Height10Percentile': group['Height'].quantile(0.1),
            'HeightSkew': group['Height'].skew(),
            'HeightKurt': group['Height'].kurtosis(),
            'UniqueAlleles': group['Allele'].nunique(),
            'RareAlleles': group['IsRare'].sum(),
        }
        
        marker_stats = group.groupby('Marker').agg({
            'Height': ['count', 'sum', 'mean', 'std', 'median', 'max'],
        }).reset_index(drop=True)
        
        if not marker_stats.empty:
            features.update({
                'PeakCountMean': marker_stats['Height']['count'].mean(),
                'PeakCountStd': marker_stats['Height']['count'].std(),
                'HeightSumMean': marker_stats['Height']['sum'].mean(),
                'HeightSumStd': marker_stats['Height']['sum'].std(),
            })
        
        imbalance_ratios = []
        for _, marker_group in group.groupby('Marker'):
            if len(marker_group) > 1:
                top_peaks = marker_group.nlargest(2, 'Height')
                ratio = top_peaks['Height'].min() / (top_peaks['Height'].max() + 1e-5)
                imbalance_ratios.append(ratio)
        
        if imbalance_ratios:
            features.update({
                'MeanImbalanceRatio': np.mean(imbalance_ratios),
                'MinImbalanceRatio': np.min(imbalance_ratios),
                'MaxImbalanceRatio': np.max(imbalance_ratios),
            })
        
        sample_features.append(features)
    
    return pd.DataFrame(sample_features)

def solve_problem1():
    print("="*50)
    print("问题1：贡献者人数识别")
    print("="*50)
    
    try:
        # 只加载附件1的数据
        df = load_processed_data(1)
        metadata = load_metadata()
        if df is None or metadata is None:
            print("错误：无法加载数据或元数据")
            return
        
        # 提取特征
        features = extract_features(df, metadata)
        
        # 检查类别分布
        class_counts = features['TrueNOC'].value_counts()
        print("类别分布:\n", class_counts)
        
        if len(class_counts) == 1:
            print(f"警告：所有样本都属于同一类别（{class_counts.index[0]}人），无法进行有效分类")
            return
        
        # 准备训练数据
        X = features.drop(['Sample File', 'TrueNOC'], axis=1).fillna(0)
        y = features['TrueNOC']
        
        # 特征选择
        selector_var = VarianceThreshold(threshold=0.01)
        X_filtered = selector_var.fit_transform(X)
        
        mi_scores = mutual_info_classif(X_filtered, y)
        mi_selected = np.argsort(mi_scores)[-20:]
        X_selected = X_filtered[:, mi_selected]
        
        # 添加多项式特征
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_selected)
        
        # 处理类别不平衡
        smote = BorderlineSMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_poly, y)
        
        # 标准化
        scaler = RobustScaler()
        X_res_scaled = scaler.fit_transform(X_res)
        
        # PCA降维
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_res_scaled)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y_res, test_size=0.2, random_state=42, stratify=y_res
        )
        
        # 模型堆叠
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42)),
            ('svm', SVC(C=5, kernel='rbf', probability=True, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42))
        ]
        
        final_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(np.unique(y_res)),
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            random_state=42
        )
        
        stacked_model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_model,
            cv=5,
            stack_method='predict_proba'
        )
        
        # 训练模型
        stacked_model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = stacked_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"测试集准确率: {acc:.4f}")
        
        # 保存结果
        results = {
            'accuracy': acc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'class_distribution': class_counts.to_dict(),
            'feature_importance': dict(zip(X.columns[mi_selected], mi_scores[mi_selected]))
        }
        
        with open(RESULTS_DIR / "problem1_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\n问题1处理完成！结果已保存")
    
    except Exception as e:
        print(f"问题1处理失败: {str(e)}")

if __name__ == "__main__":
    solve_problem1()