import pandas as pd
import numpy as np

def load_and_process_data(file_path):
    """加载和预处理STR数据"""
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        print(f"STR数据加载完成: {data.shape}")
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise

def get_contributor_num_and_ratio(sample_name):
    """从样本名称中提取贡献者数量和比例（备用函数）"""
    try:
        parts = sample_name.split('_')
        if len(parts) < 2:
            raise ValueError("样本名称格式不正确")
        
        info_part = parts[1]
        segments = info_part.split('-')
        
        if len(segments) < 4:
            raise ValueError("样本信息格式不正确")
        
        # 贡献者数量
        contributor_part = segments[2]
        contributor_ids = contributor_part.split('_')
        n_contributors = len(contributor_ids)
        
        # 比例
        ratio_part = segments[3]
        ratios = [int(x) for x in ratio_part.split(';') if x.isdigit()]
        
        if len(ratios) != n_contributors:
            raise ValueError("贡献者数量与比例数量不匹配")
        
        return n_contributors, ratios
        
    except Exception as e:
        raise ValueError(f"解析样本名称失败: {e}")