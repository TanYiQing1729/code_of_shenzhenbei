import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
def get_contributor_num_and_ratio(sample):
    """
    从Sample File字符串中提取贡献者人数和比例
    返回：人数（int），比例（list of int，若无则全为1）
    """
    try:
        # 取第三段（下标2），如40_41-1;4 或 40_41_42_43-1;1;1;1
        part = sample.split('-')[2]
        # 取编号部分
        ids_part = part.split(';')[0]
        id_list = ids_part.split('_')
        num = len(id_list)
        # 取比例部分
        ratio_part = part.split(';')[1:]
        ratios = [int(x) for x in ratio_part if x.isdigit()]
        # 如果没有比例，默认全为1
        if not ratios or len(ratios) != num:
            ratios = [1] * num
        return num, ratios
    except Exception as e:
        print("解析人数/比例失败:", sample, e)
        return 1, [1]

# 提取特征
def extract_features(df, marker_list=None):
    features = []
    sample_files = []
    labels = []
    ratios_list = []
    for sample, group in df.groupby('Sample File'):
        sample_files.append(sample)
        n, ratios = get_contributor_num_and_ratio(sample)
        labels.append(n)
        ratios_list.append(ratios)
        feature = {}
        feature['contributor_count'] = n
        feature['allele_total'] = 0
        feature['non_ol_allele_total'] = 0
        feature['height_total'] = 0
        feature['ol_allele_total'] = 0
        feature['marker_diversity'] = 0
        for marker in marker_list:
            mgroup = group[group['Marker'] == marker]
            alleles = []
            heights = []
            for i in range(1, 32):
                allele = mgroup[f'Allele {i}'].values[0] if not mgroup.empty else np.nan
                height = mgroup[f'Height {i}'].values[0] if not mgroup.empty else np.nan
                if pd.isna(allele) or pd.isna(height) or float(height) <= 0:
                    continue
                alleles.append(str(allele))
                heights.append(float(height))
            # marker级统计
            feature[f'{marker}_allele_count'] = len(alleles)
            feature[f'{marker}_unique_allele_count'] = len(set(alleles))
            feature[f'{marker}_allele_diversity'] = len(set(alleles)) / (len(alleles)+1e-6)
            feature[f'{marker}_height_mean'] = np.mean(heights) if heights else 0
            feature[f'{marker}_height_std'] = np.std(heights) if len(heights) > 1 else 0
            feature[f'{marker}_height_max'] = np.max(heights) if heights else 0
            feature[f'{marker}_height_min'] = np.min(heights) if heights else 0
            feature[f'{marker}_height_sum'] = np.sum(heights) if heights else 0
            feature[f'{marker}_height_range'] = feature[f'{marker}_height_max'] - feature[f'{marker}_height_min'] if heights else 0
            feature[f'{marker}_height_cv'] = feature[f'{marker}_height_std'] / feature[f'{marker}_height_mean'] if feature[f'{marker}_height_mean'] > 0 else 0
            # topN峰高
            sorted_heights = sorted(heights, reverse=True)
            for k in range(4):
                feature[f'{marker}_height_top{k+1}'] = sorted_heights[k] if len(sorted_heights) > k else 0
            # topN比值
            feature[f'{marker}_top1_top2_ratio'] = sorted_heights[0] / sorted_heights[1] if len(sorted_heights) > 1 and sorted_heights[1] > 0 else 0
            feature[f'{marker}_top2_top3_ratio'] = sorted_heights[1] / sorted_heights[2] if len(sorted_heights) > 2 and sorted_heights[2] > 0 else 0
            feature[f'{marker}_top3_top4_ratio'] = sorted_heights[2] / sorted_heights[3] if len(sorted_heights) > 3 and sorted_heights[3] > 0 else 0
            feature[f'{marker}_top_to_mean_ratio'] = sorted_heights[0] / np.mean(sorted_heights) if len(sorted_heights) > 0 and np.mean(sorted_heights) > 0 else 0
            # 偏度和峰度
            if len(heights) > 2:
                feature[f'{marker}_height_skew'] = pd.Series(heights).skew()
                feature[f'{marker}_height_kurt'] = pd.Series(heights).kurt()
            else:
                feature[f'{marker}_height_skew'] = 0
                feature[f'{marker}_height_kurt'] = 0
            # OL等位基因
            ol_count = sum(1 for a in alleles if a == 'OL')
            feature[f'{marker}_ol_allele_count'] = ol_count
            feature[f'{marker}_ol_allele_ratio'] = ol_count / len(alleles) if len(alleles) > 0 else 0
            # 超2/4/6/8等位基因
            feature[f'{marker}_exceeds_2_alleles'] = 1 if len(alleles) > 2 else 0
            feature[f'{marker}_exceeds_4_alleles'] = 1 if len(alleles) > 4 else 0
            feature[f'{marker}_exceeds_6_alleles'] = 1 if len(alleles) > 6 else 0
            feature[f'{marker}_exceeds_8_alleles'] = 1 if len(alleles) > 8 else 0
            # 预期等位基因比
            expected_alleles = 2 * n
            feature[f'{marker}_expected_allele_ratio'] = len(alleles) / expected_alleles if expected_alleles > 0 else 0
            # 全局统计
            feature['allele_total'] += len(alleles)
            feature['non_ol_allele_total'] += len([a for a in alleles if a != 'OL'])
            feature['height_total'] += np.sum(heights) if heights else 0
            feature['ol_allele_total'] += ol_count
            feature['marker_diversity'] += feature[f'{marker}_allele_diversity']
        # 全局比例
        feature['ol_allele_ratio_total'] = feature['ol_allele_total'] / feature['allele_total'] if feature['allele_total'] > 0 else 0
        feature['marker_diversity'] = feature['marker_diversity'] / (len(marker_list) if marker_list else 1)
        features.append(feature)
    X = pd.DataFrame(features).fillna(0).values
    y = np.array(labels)

    df_feat = pd.DataFrame(features)
    # 1. 异常值处理
    for col in df_feat.columns:
        if 'height' in col:
            upper = df_feat[col].quantile(0.99)
            df_feat[col] = np.where(df_feat[col] > upper, upper, df_feat[col])
    # 2. 对数变换
    for col in df_feat.columns:
        if 'height' in col and (df_feat[col] > 0).all():
            df_feat[col] = np.log1p(df_feat[col])
    # 3. 缺失值处理
    df_feat = df_feat.fillna(df_feat.mean())
    # 4. 去掉方差极小特征
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=1e-5)
    X = selector.fit_transform(df_feat)
    #优化
    return X, y, sample_files, ratios_list

class STRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
