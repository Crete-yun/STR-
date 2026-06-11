import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到Python路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from src.utils import (
        DATA_DIR, PROCESSED_DIR, FEATURES_DIR,
        MODEL_DIR, RESULTS_DIR, validate_data_integrity
    )
    from src.preprocessing import STRDataPreprocessor
    from src.problem1 import solve_problem1
    from src.problem2 import solve_problem2
    from src.problem3 import solve_problem3
    from src.problem4 import solve_problem4
except ImportError as e:
    print(f"❌ 导入错误: {str(e)}")
    print("请检查：")
    print("1. 确认存在 src/ 目录且包含 __init__.py 文件")
    print("2. 确认所有.py文件都在正确位置")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="法医STR分析系统")
    parser.add_argument('--preprocess', action='store_true', help="运行数据预处理")
    parser.add_argument('--problem1', action='store_true', help="解决贡献者人数识别问题")
    parser.add_argument('--problem2', action='store_true', help="解决贡献者比例估计问题")
    parser.add_argument('--problem3', action='store_true', help="解决贡献者基因型推断问题")
    parser.add_argument('--problem4', action='store_true', help="解决噪声抑制问题")
    parser.add_argument('--all', action='store_true', help="运行所有处理步骤")
    parser.add_argument('--format', default='xlsx', choices=['csv', 'xlsx'],
                      help="输入文件格式 (默认: xlsx)")
    
    args = parser.parse_args()
    
    if not any([args.preprocess, args.problem1, args.problem2, args.problem3, args.problem4, args.all]):
        args.all = True
    
    for d in [PROCESSED_DIR, FEATURES_DIR, MODEL_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    run_preprocess = args.preprocess or args.all
    if any([args.problem1, args.problem2, args.problem3, args.problem4]) and not run_preprocess:
        if not validate_data_integrity():
            print(f"[{datetime.now()}] 数据完整性验证失败，自动运行预处理...")
            preprocessor = STRDataPreprocessor()
            preprocessor.run_preprocessing(file_format=args.format)
            run_preprocess = True
    
    if run_preprocess:
        print("="*50)
        print(f"[{datetime.now()}] 开始数据预处理")
        print("="*50)
        preprocessor = STRDataPreprocessor()
        success = preprocessor.run_preprocessing(file_format=args.format)
        
        # 即使完整性检查失败也继续运行
        if not success:
            print(f"[{datetime.now()}] 预处理完成但有警告，将继续运行问题处理")
        elif not validate_data_integrity():
            print(f"[{datetime.now()}] 数据完整性验证失败，但将继续运行问题处理")
    
    if args.problem1 or args.all:
        print("="*50)
        print(f"[{datetime.now()}] 开始解决问题1：贡献者人数识别")
        print("="*50)
        solve_problem1()
    
    if args.problem2 or args.all:
        print("="*50)
        print(f"[{datetime.now()}] 开始解决问题2：贡献者比例估计")
        print("="*50)
        solve_problem2()
    
    if args.problem3 or args.all:
        print("="*50)
        print(f"[{datetime.now()}] 开始解决问题3：贡献者基因型推断")
        print("="*50)
        solve_problem3()
    
    if args.problem4 or args.all:
        print("="*50)
        print(f"[{datetime.now()}] 开始解决问题4：噪声抑制")
        print("="*50)
        solve_problem4()
    
    print(f"[{datetime.now()}] 所有处理完成！")

if __name__ == "__main__":
    main()