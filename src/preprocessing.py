import pandas as pd
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
from src.utils import (
    RAW_DIR, PROCESSED_DIR, FEATURES_DIR, GENOTYPES_DIR,
    load_data_file, save_metadata
)

class STRDataPreprocessor:
    def __init__(self, min_peak_height=50, stutter_ratio=0.15, rare_allele_threshold=0.05):
        self.min_peak_height = min_peak_height
        self.stutter_ratio = stutter_ratio
        self.rare_allele_threshold = rare_allele_threshold
        self.allele_freq = defaultdict(lambda: defaultdict(int))
        self.total_samples = 0
        # 更新标记列表以匹配附件3的列名
        self.marker_info = {
            'D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358', 
            'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433',
            'vWA', 'TPOX', 'D18S51', 'AM', 'D5S818', 'FGA'  # 注意：AM而不是AMEL
        }

    def _parse_sample_id(self, sample_file):
            """解析样本ID，提取贡献者人数、ID和比例"""
            try:
                # 尝试解析新格式（RD14-0003开头）
                if "RD14-0003-" in sample_file:
                    stem = Path(sample_file).stem  # 移除文件扩展名
                    key = "RD14-0003-"
                    idx = stem.find(key)
                    if idx != -1:
                        # 提取关键部分（RD14-0003-之后的内容）
                        substr = stem[idx + len(key):]
                        parts = substr.split('-', 2)  # 分割成3部分
                        if len(parts) >= 2:
                            # 解析贡献者ID和比例
                            contributors = parts[0].split('_')
                            ratios = [float(r) for r in parts[1].split(';') if r]
                    
                            # 归一化比例
                            total = sum(ratios)
                            normalized_ratios = [r/total for r in ratios]
                            return len(contributors), contributors, normalized_ratios
        
                # 旧格式解析（原始逻辑）
                parts = sample_file.split('-')
                if len(parts) < 3:
                    return 1, [], []
        
                contrib_part = parts[2]
                if '_' in contrib_part and '-' in contrib_part:
                    subparts = contrib_part.split('-', 1)
                    if len(subparts) == 2:
                        contributors = subparts[0].split('_')
                        ratios = [float(r) for r in subparts[1].split(';') if r]
                        total = sum(ratios)
                        normalized_ratios = [r/total for r in ratios]
                        return len(contributors), contributors, normalized_ratios
                return 1, [], []
            except Exception as e:
                print(f"警告：样本 {sample_file} 解析失败 - {str(e)}")
                return 1, [], []

    def _process_row(self, row):
        """处理单行数据，提取所有等位基因信息"""
        alleles = []
        for i in range(1, 6):  # 通常不超过5个等位基因
            allele_col = f'Allele {i}'
            size_col = f'Size {i}'
            height_col = f'Height {i}'
            
            if allele_col in row and pd.notna(row[allele_col]) and str(row[allele_col]).strip() not in ['', 'OL']:
                allele_value = str(row[allele_col]).strip()
                # 特殊处理AMEL基因座
                if row['Marker'] == 'AMEL' and allele_value in ['X', 'Y']:
                    alleles.append({
                        'Marker': row['Marker'],
                        'Dye': row['Dye'],
                        'Allele': allele_value,
                        'Size': row[size_col] if size_col in row else np.nan,
                        'Height': row[height_col] if height_col in row else np.nan,
                    })
                elif row['Marker'] != 'AMEL':
                    alleles.append({
                        'Marker': row['Marker'],
                        'Dye': row['Dye'],
                        'Allele': allele_value,
                        'Size': row[size_col] if size_col in row else np.nan,
                        'Height': row[height_col] if height_col in row else np.nan,
                    })
        return alleles


    def _detect_stutter(self, locus_df):
        """改进的stutter峰检测"""
        try:
            if len(locus_df) < 2:
                return locus_df
            
            if 'IsStutter' not in locus_df.columns:
                locus_df['IsStutter'] = False
            
            sorted_peaks = locus_df.sort_values('Size')
            stutter_flags = [False] * len(locus_df)
    
            for i in range(len(sorted_peaks)):
                peak = sorted_peaks.iloc[i]
                if peak['IsStutter']:
                    continue
                    
                for j in range(i+1, len(sorted_peaks)):
                    other_peak = sorted_peaks.iloc[j]
                    size_diff = other_peak['Size'] - peak['Size']
                    
                    # 检测-1和-4 stutter峰
                    if abs(size_diff - (-4)) < 0.5 or abs(size_diff - (-1)) < 0.5:
                        height_ratio = other_peak['Height'] / peak['Height']
                        if height_ratio <= self.stutter_ratio:
                            stutter_flags[j] = True
                    elif size_diff > 1:
                        break
    
            locus_df['IsStutter'] = stutter_flags
            return locus_df
        except Exception as e:
            print(f"Stutter检测出错: {str(e)}")
            return locus_df

    def _calculate_allele_frequency(self, df):
        """计算等位基因频率"""
        for marker, group in df.groupby('Marker'):
            for allele in group['Allele']:
                self.allele_freq[marker][allele] += 1
        self.total_samples = df['Sample File'].nunique()

    def _get_rare_alleles(self):
        """获取稀有等位基因"""
        rare_alleles = {}
        for marker, alleles in self.allele_freq.items():
            total = sum(alleles.values())
            rare_alleles[marker] = {
                allele for allele, count in alleles.items() 
                if count / total < self.rare_allele_threshold
            }
        return rare_alleles

    def preprocess_data(self, df, attachment_num):
        """预处理数据"""
        processed_data = []
        sample_metadata = {}

        df.columns = df.columns.str.strip()
        sample_file_col = next((col for col in df.columns if 'sample' in col.lower()), None)
        
        if sample_file_col is None:
            print("错误: 数据中找不到样本文件列")
            return pd.DataFrame(), {}

        df.rename(columns={sample_file_col: 'Sample File'}, inplace=True)

        for _, row in df.iterrows():
            try:
                sample_file = row['Sample File']
                if pd.isna(sample_file):
                    continue
            
                contributor_count, contributors, ratios = self._parse_sample_id(sample_file)
        
                if sample_file not in sample_metadata:
                    sample_metadata[sample_file] = {
                        'attachment': attachment_num,
                        'contributor_count': contributor_count,
                        'contributors': contributors,
                        'ratio': ratios,
                        'markers': set()
                    }
        
                alleles = self._process_row(row)
                if not alleles:
                    continue
            
                marker_df = pd.DataFrame(alleles)
                sample_metadata[sample_file]['markers'].add(row['Marker'])
        
                marker_df['IsStutter'] = False
                if len(marker_df) > 1:
                    marker_df = self._detect_stutter(marker_df)
        
                filtered = marker_df.loc[
                    (marker_df['Height'] >= self.min_peak_height) &
                    (~marker_df['IsStutter'])
                ].copy()
        
                if not filtered.empty:
                    filtered['Sample File'] = sample_file
                    processed_data.append(filtered)
            
            except Exception as e:
                print(f"处理样本时出错: {str(e)}")
                continue

        full_df = pd.concat(processed_data, ignore_index=True) if processed_data else pd.DataFrame()

        if not full_df.empty:
            self._calculate_allele_frequency(full_df)
            rare_alleles = self._get_rare_alleles()
            full_df['IsRare'] = full_df.apply(
                lambda row: row['Allele'] in rare_alleles.get(row['Marker'], set()), 
                axis=1
            )

        return full_df, sample_metadata

    def preprocess_attachment3(self, df, attachment_num):
        """专门处理附件3的基因型数据格式"""
        processed_data = []
        sample_metadata = {}
        
        # 检查必需列
        required_columns = ['Reseach ID', 'Sample ID']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"附件3缺少必要列: {', '.join(missing_cols)}")
            return pd.DataFrame(), {}
        
        # 重命名AM列为AMEL以保持一致性
        if 'AM' in df.columns:
            df = df.rename(columns={'AM': 'AMEL'})
            self.marker_info.add('AMEL')
            self.marker_info.discard('AM')
        
        for _, row in df.iterrows():
            sample_id = f"{row['Reseach ID']}-{row['Sample ID']}"
            
            if sample_id not in sample_metadata:
                sample_metadata[sample_id] = {
                    'attachment': attachment_num,
                    'contributor_count': 1,
                    'contributors': [str(row['Sample ID'])],  # 确保字符串类型
                    'ratio': [1.0],
                    'markers': []  # 改为列表而不是set
                }
            
            for marker in self.marker_info:
                if marker not in row:
                    continue
                    
                # 处理可能的NaN值
                marker_value = row[marker]
                if pd.isna(marker_value):
                    continue
                    
                alleles = str(marker_value).split(',')
                alleles = [a.strip() for a in alleles if a.strip()]
                
                if len(alleles) < 1:
                    continue
                    
                # 特殊处理AMEL基因座
                if marker == 'AMEL':
                    # 将AMEL值转换为标准格式
                    if 'X' in marker_value or 'Y' in marker_value:
                        alleles = [a for a in alleles if a in ['X', 'Y']]
                    else:
                        # 处理数字格式的AMEL值
                        alleles = ['X' if float(a) < 100 else 'Y' for a in alleles]
                
                for allele in alleles:
                    processed_data.append({
                        'Sample File': sample_id,
                        'Marker': 'AMEL' if marker == 'AMEL' else marker,
                        'Allele': allele,
                        'Height': 1000,
                        'Size': 0,
                        'IsStutter': False,
                        'IsRare': False
                    })
                
                # 添加到markers列表
                marker_name = 'AMEL' if marker == 'AMEL' else marker
                if marker_name not in sample_metadata[sample_id]['markers']:
                    sample_metadata[sample_id]['markers'].append(marker_name)
        
        full_df = pd.DataFrame(processed_data) if processed_data else pd.DataFrame()
        
        if not full_df.empty:
            self._calculate_allele_frequency(full_df)
            rare_alleles = self._get_rare_alleles()
            full_df['IsRare'] = full_df.apply(
                lambda row: row['Allele'] in rare_alleles.get(row['Marker'], set()), 
                axis=1
            )
        
        return full_df, sample_metadata

    def run_preprocessing(self, file_format='xlsx'):
        """执行预处理并保存所有样本"""
        print("开始预处理...")

        all_processed_data = []
        all_sample_metadata = {}

        attachment_info = {
            1: {'filename': 'attachment1.xlsx', 'expected_samples': 816, 'processor': 'default'},
            2: {'filename': 'attachment2.xlsx', 'expected_samples': 801, 'processor': 'default'},
            3: {'filename': 'attachment3.xlsx', 'expected_samples': 50, 'processor': 'attachment3'},
            4: {'filename': 'attachment4.xlsx', 'expected_samples': 816, 'processor': 'default'}
        }

        for attachment_num, info in attachment_info.items():
            filename = info['filename']
            expected_samples = info['expected_samples']
            
            print(f"\n处理附件{attachment_num} ({filename})...")
            df = load_data_file(filename, file_format)

            if df is None:
                print(f"附件{attachment_num}数据加载失败，跳过")
                continue
            
            if len(df) < expected_samples * 0.9:
                print(f"警告: 附件{attachment_num}样本量不足，预期{expected_samples}，实际{len(df)}")

            if info['processor'] == 'attachment3':
                processed_data, sample_metadata = self.preprocess_attachment3(df, attachment_num)
            else:
                required_columns = ['Sample File', 'Marker', 'Allele 1', 'Height 1']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"附件{attachment_num}缺少必要列: {', '.join(missing_cols)}")
                    continue
                
                processed_data, sample_metadata = self.preprocess_data(df, attachment_num)

            if not processed_data.empty:
                processed_data.to_csv(PROCESSED_DIR / f"processed_attachment{attachment_num}.csv", index=False)
                all_processed_data.append(processed_data)
                all_sample_metadata.update(sample_metadata)
            
                print(f"附件{attachment_num}处理完成，样本数: {len(sample_metadata)}")
            else:
                print(f"附件{attachment_num}无有效数据")

        if all_sample_metadata:
            save_metadata(all_sample_metadata)
            print(f"\n预处理完成！总样本数: {len(all_sample_metadata)}")
            return True
        else:
            print("预处理失败，无有效数据")
            return False