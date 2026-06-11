# utils.py
import os
import json
import pandas as pd
import joblib
from pathlib import Path
import logging

# 使用Pathlib设置路径
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
GENOTYPES_DIR = DATA_DIR / "genotypes"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# 确保目录存在
for d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, GENOTYPES_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 创建全局日志对象
logger = logging.getLogger("utils")
logger.setLevel(logging.DEBUG)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(console_handler)

def setup_file_logger():
    """配置文件日志记录器"""
    # 确保结果目录存在
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建文件处理器
    log_file = RESULTS_DIR / "project.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 添加文件处理器
    logger.addHandler(file_handler)
    return logger

def load_data_file(filename, file_format='xlsx'):
    """加载数据文件"""
    path = RAW_DIR / filename
    if not path.exists():
        logger.error(f"文件不存在 - {path}")
        return None
        
    try:
        if file_format == 'xlsx':
            df = pd.read_excel(path, engine='openpyxl')
        else:
            df = pd.read_csv(path)
        
        logger.info(f"成功加载 {filename}, 形状: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"文件加载失败: {path} | 错误: {str(e)}")
        return None

def load_metadata():
    """加载元数据"""
    metadata_path = PROCESSED_DIR / "sample_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"元数据加载失败: {str(e)}")
            return {}
    return {}

def save_metadata(metadata):
    """保存元数据（忽略序列化错误）"""
    metadata_path = PROCESSED_DIR / "sample_metadata.json"
    try:
        # 尝试序列化所有内容为字符串
        serializable_metadata = {}
        for sample, data in metadata.items():
            serializable_data = {}
            for key, value in data.items():
                try:
                    # 尝试序列化值
                    json.dumps(value)
                    serializable_data[key] = value
                except TypeError:
                    # 如果序列化失败，转换为字符串
                    serializable_data[key] = str(value)
            serializable_metadata[sample] = serializable_data
        
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"元数据保存失败: {str(e)}")
        return False

def load_processed_data(attachment):
    """加载预处理数据"""
    path = PROCESSED_DIR / f"processed_attachment{attachment}.csv"
    if not path.exists():
        logger.error(f"预处理数据未找到: {path}")
        return None
        
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"预处理数据加载失败: {path} | 错误: {str(e)}")
        return None

def load_genotype_data():
    """加载基因型数据（跳过错误）"""
    path = GENOTYPES_DIR / "genotype_data.csv"
    if path.exists():
        try:
            genotype_db = pd.read_csv(path)
        except Exception as e:
            logger.error(f"基因型数据加载失败: {path} | 错误: {str(e)}")
            genotype_db = pd.DataFrame()
    else:
        genotype_db = pd.DataFrame()
    
    # 尝试加载附件3的数据作为补充
    try:
        df = load_processed_data(3)
        if df is not None and not df.empty:
            attachment3_genotypes = []
            for sample_file, group in df.groupby('Sample File'):
                sample_id = sample_file.split('-')[-1]  # 提取Sample ID
                for _, row in group.iterrows():
                    attachment3_genotypes.append({
                        'Sample ID': sample_id,
                        'Marker': row['Marker'],
                        'Allele': row['Allele']
                    })
            
            attachment3_df = pd.DataFrame(attachment3_genotypes)
            genotype_db = pd.concat([genotype_db, attachment3_df], ignore_index=True)
    except Exception as e:
        logger.error(f"附件3基因型数据加载失败: {str(e)}")
    
    if genotype_db.empty:
        logger.warning("无基因型数据可用")
        return pd.DataFrame()  # 返回空DataFrame而不是None
    
    return genotype_db

def validate_data_integrity():
    """简化数据完整性检查（仅检查文件存在）"""
    expected_files = [
        "processed_attachment1.csv",
        "processed_attachment2.csv",
        "processed_attachment3.csv",
        "processed_attachment4.csv",
        "sample_metadata.json"
    ]
    
    missing_files = []
    for filename in expected_files:
        path = PROCESSED_DIR / filename
        if not path.exists():
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"缺失文件: {', '.join(missing_files)}")
        return False
    
    return True

# 初始化文件日志记录器
setup_file_logger()
logger.info("utils模块初始化完成")