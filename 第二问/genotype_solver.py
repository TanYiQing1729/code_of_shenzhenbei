import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize
import warnings
warnings.filterwarnings('ignore')

class GenotypeMixtureSolver:
    """基于基因型数据的混合比例求解器"""
    
    def __init__(self):
        self.genotype_data = None
        self.str_markers = ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358', 'TH01', 
                           'D13S317', 'D16S539', 'D2S1338', 'D19S433', 'vWA', 'TPOX', 
                           'D18S51', 'AMEL', 'D5S818', 'FGA']
        
    def load_genotype_data(self, genotype_file_path):
        """加载基因型数据"""
        try:
            # 读取数据并清理
            self.genotype_data = pd.read_csv(genotype_file_path, encoding='utf-8')
            
            # 清理数据：删除空行和无效行
            self.genotype_data = self.genotype_data.dropna(subset=['Reseach ID', 'Sample ID'])
            
            # 确保Sample ID为数值类型
            self.genotype_data['Sample ID'] = pd.to_numeric(self.genotype_data['Sample ID'], errors='coerce')
            self.genotype_data = self.genotype_data.dropna(subset=['Sample ID'])
            self.genotype_data['Sample ID'] = self.genotype_data['Sample ID'].astype(int)
            
            print(f"基因型数据加载成功:")
            print(f"  数据形状: {self.genotype_data.shape}")
            print(f"  列名: {list(self.genotype_data.columns)}")
            
            # 查看数据结构
            if 'Reseach ID' in self.genotype_data.columns:
                unique_research_ids = self.genotype_data['Reseach ID'].unique()
                print(f"  研究ID: {unique_research_ids}")
            
            if 'Sample ID' in self.genotype_data.columns:
                sample_ids = sorted(self.genotype_data['Sample ID'].unique())
                print(f"  样本ID范围: {min(sample_ids)}-{max(sample_ids)} ({len(sample_ids)}个)")
                
            return True
        except Exception as e:
            print(f"加载基因型数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def parse_sample_name(self, sample_name):
        """解析样本名称，提取贡献者ID和真实比例"""
        try:
            print(f"  解析样本名称: {sample_name}")
            
            # 样本名称格式分析：
            # E04_RD14-0003-36_37_38_39-1;2;2;1-M3a-0.186IP-Q0.7_001.5sec.fsa
            # F03_RD14-0003-49_50_29-1;4;1-M3S30-0.09IP-Q2.9_002.5sec.fsa
            # B02_RD14-0003-42_43-1;9-M1S10-0.75IP-Q1.2_002.5sec.fsa
            
            # 移除文件扩展名
            name_without_ext = sample_name.replace('.fsa', '')
            print(f"    无扩展名: {name_without_ext}")
            
            # 用正则表达式来匹配模式
            import re
            
            # 匹配模式: 开头_RD14-0003-数字序列-比例序列-其他
            pattern = r'^[A-Z]\d+_RD14-0003-([0-9_]+)-([0-9;]+)-'
            match = re.match(pattern, name_without_ext)
            
            if match:
                contributor_numbers = match.group(1)  # 例如: "36_37_38_39"
                ratio_numbers = match.group(2)        # 例如: "1;2;2;1"
                
                print(f"    匹配成功:")
                print(f"      贡献者编号: {contributor_numbers}")
                print(f"      比例编号: {ratio_numbers}")
                
                # 解析贡献者ID
                contributor_ids = contributor_numbers.split('_')
                sample_ids = [int(cid) for cid in contributor_ids]
                
                # 解析比例
                ratios = [int(x) for x in ratio_numbers.split(';') if x.isdigit()]
                
                print(f"    提取的贡献者Sample ID: {sample_ids}")
                print(f"    提取的比例: {ratios}")
                
                if len(sample_ids) == len(ratios):
                    return sample_ids, ratios
                else:
                    print(f"    警告: 贡献者数量({len(sample_ids)})与比例数量({len(ratios)})不匹配")
                    return None, None
            else:
                print(f"    错误: 样本名称不匹配预期模式")
                
                # 尝试备用解析方法
                print(f"    尝试备用解析方法...")
                
                # 按 '_' 分割并重新组合
                parts = name_without_ext.split('_')
                if len(parts) >= 2:
                    # 从第二个部分开始，找到最后一个以时间结尾的部分
                    info_parts = parts[1:-1]  # 排除第一个和最后一个部分
                    info_string = '_'.join(info_parts)
                    
                    print(f"    信息字符串: {info_string}")
                    
                    # 再次尝试用正则表达式匹配
                    pattern2 = r'RD14-0003-([0-9_]+)-([0-9;]+)'
                    match2 = re.search(pattern2, info_string)
                    
                    if match2:
                        contributor_numbers = match2.group(1)
                        ratio_numbers = match2.group(2)
                        
                        print(f"    备用方法匹配成功:")
                        print(f"      贡献者编号: {contributor_numbers}")
                        print(f"      比例编号: {ratio_numbers}")
                        
                        # 解析贡献者ID
                        contributor_ids = contributor_numbers.split('_')
                        sample_ids = [int(cid) for cid in contributor_ids]
                        
                        # 解析比例
                        ratios = [int(x) for x in ratio_numbers.split(';') if x.isdigit()]
                        
                        print(f"    提取的贡献者Sample ID: {sample_ids}")
                        print(f"    提取的比例: {ratios}")
                        
                        if len(sample_ids) == len(ratios):
                            return sample_ids, ratios
                        else:
                            print(f"    警告: 贡献者数量({len(sample_ids)})与比例数量({len(ratios)})不匹配")
                            return None, None
                
                return None, None
            
        except Exception as e:
            print(f"解析样本名称失败 {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_contributor_genotype(self, sample_id, marker):
        """获取指定贡献者在指定基因座的基因型"""
        if self.genotype_data is None:
            return None
        
        # 查找基因型数据
        mask = (self.genotype_data['Sample ID'] == sample_id)
        contributor_data = self.genotype_data[mask]
        
        if contributor_data.empty:
            print(f"    警告: 未找到样本ID {sample_id} 的基因型数据")
            return None
        
        # 获取该基因座的基因型
        if marker in contributor_data.columns:
            genotype_str = contributor_data[marker].iloc[0]
            
            if pd.isna(genotype_str) or str(genotype_str) == 'nan':
                return None
            
            # 解析基因型字符串，例如: "12,15" -> ["12", "15"]
            genotype_str = str(genotype_str).replace('"', '')  # 移除引号
            alleles = [allele.strip() for allele in genotype_str.split(',')]
            alleles = [allele for allele in alleles if allele and allele != 'nan']
            
            return alleles if alleles else None
        
        return None
    
    def extract_peak_data(self, sample_data, marker):
        """从样本数据中提取指定基因座的峰高信息"""
        marker_data = sample_data[sample_data['Marker'] == marker]
        if marker_data.empty:
            return [], []
        
        heights = []
        alleles = []
        
        # 提取所有峰的信息
        for i in range(1, 101):  # 最多100个峰
            height_col = f'Height {i}'
            allele_col = f'Allele {i}'
            
            if height_col in marker_data.columns and allele_col in marker_data.columns:
                height = marker_data[height_col].iloc[0]
                allele = marker_data[allele_col].iloc[0]
                
                if not pd.isna(height) and not pd.isna(allele) and height > 0:
                    # 过滤掉 "OL" (over-limit) 等非数字等位基因
                    allele_str = str(allele).strip()
                    if allele_str != "OL" and allele_str != "nan":
                        try:
                            # 验证是否为有效的等位基因名称（可能包含小数点）
                            float(allele_str)  # 如果能转换为数字，说明是有效的等位基因
                            heights.append(float(height))
                            alleles.append(allele_str)
                        except ValueError:
                            # 跳过非数字的等位基因
                            continue
        
        if not heights:
            return [], []
        
        # 峰高阈值过滤（相对阈值）
        heights = np.array(heights)
        max_height = np.max(heights)
        threshold = max_height * 0.06  # 6%阈值
        
        valid_indices = heights >= threshold
        filtered_heights = heights[valid_indices]
        filtered_alleles = [alleles[i] for i in range(len(alleles)) if valid_indices[i]]
        
        return filtered_heights.tolist(), filtered_alleles
    
    def build_genotype_matrix(self, sample_ids, marker, observed_alleles):
        """构建基因型矩阵G，其中G[i,j]表示贡献者j对等位基因i的贡献"""
        n_alleles = len(observed_alleles)
        n_contributors = len(sample_ids)
        
        G = np.zeros((n_alleles, n_contributors))
        
        for j, sample_id in enumerate(sample_ids):
            contributor_alleles = self.get_contributor_genotype(sample_id, marker)
            
            if contributor_alleles is None:
                print(f"    警告: 未获取到样本ID {sample_id} 在 {marker} 的基因型")
                continue
            
            print(f"    样本ID {sample_id} 在 {marker} 的基因型: {contributor_alleles}")
            
            # 计算每个观测等位基因在该贡献者中的出现次数
            for i, observed_allele in enumerate(observed_alleles):
                count = contributor_alleles.count(observed_allele)
                G[i, j] = count
        
        return G
    # 在现有代码基础上添加缺失的函数

    def solve_single_marker(self, sample_data, sample_ids, marker, method='nnls'):
        """求解单个基因座的混合比例"""
        print(f"    分析基因座: {marker}")
        
        # 1. 提取峰高数据
        heights, alleles = self.extract_peak_data(sample_data, marker)
        
        if len(heights) == 0 or len(alleles) == 0:
            print(f"      无有效峰高数据")
            return None, f"无有效峰高数据"
        
        print(f"      观测到的等位基因: {alleles}")
        print(f"      对应峰高: {[f'{h:.0f}' for h in heights]}")
        
        # 2. 构建基因型矩阵
        G = self.build_genotype_matrix(sample_ids, marker, alleles)
        
        if G.sum() == 0:
            print(f"      基因型矩阵为空")
            return None, f"基因型矩阵为空"
        
        print(f"      基因型矩阵 G 形状: {G.shape}")
        print(f"      基因型矩阵 G:\n{G}")
        
        # 3. 求解线性方程 G @ r = h
        h = np.array(heights)
        n_alleles, n_contributors = G.shape
        
        # 检查维度匹配
        if n_alleles != len(h):
            print(f"      错误: 基因型矩阵行数({n_alleles})与峰高数量({len(h)})不匹配")
            return None, f"维度不匹配"
        
        try:
            if method == 'nnls':
                # 求解 G @ r = h
                ratios, residual = nnls(G, h)
            elif method == 'lstsq':
                # 普通最小二乘
                ratios, residuals, rank, s = np.linalg.lstsq(G, h, rcond=None)
                ratios = np.maximum(ratios, 0)  # 确保非负
            else:
                return None, f"未知求解方法: {method}"
            
            # 4. 归一化
            if ratios.sum() > 0:
                ratios_normalized = ratios / ratios.sum()
                print(f"      求解结果: {[f'{x:.3f}' for x in ratios_normalized]}")
                return ratios_normalized, None
            else:
                return None, "所有比例为零"
                
        except Exception as e:
            print(f"      求解失败: {e}")
            # 尝试备用方法：伪逆
            try:
                print(f"      尝试伪逆方法...")
                G_pinv = np.linalg.pinv(G)
                ratios = G_pinv @ h
                ratios = np.maximum(ratios, 0)  # 确保非负
                
                if ratios.sum() > 0:
                    ratios_normalized = ratios / ratios.sum()
                    print(f"      伪逆求解结果: {[f'{x:.3f}' for x in ratios_normalized]}")
                    return ratios_normalized, None
                else:
                    return None, "所有比例为零"
            except Exception as e2:
                return None, f"求解失败: {e2}"

    def solve_single_marker_robust(self, sample_data, sample_ids, marker):
        """稳健的单基因座求解方法 - 使用多种方法组合"""
        print(f"    分析基因座: {marker} (稳健模式)")
        
        # 1. 提取峰高数据
        heights, alleles = self.extract_peak_data(sample_data, marker)
        
        if len(heights) == 0 or len(alleles) == 0:
            print(f"      无有效峰高数据")
            return None, f"无有效峰高数据"
        
        print(f"      观测到的等位基因: {alleles}")
        print(f"      对应峰高: {[f'{h:.0f}' for h in heights]}")
        
        # 2. 构建基因型矩阵
        G = self.build_genotype_matrix(sample_ids, marker, alleles)
        
        if G.sum() == 0:
            print(f"      基因型矩阵为空")
            return None, f"基因型矩阵为空"
        
        print(f"      基因型矩阵 G 形状: {G.shape}")
        print(f"      基因型矩阵 G:\n{G}")
        
        # 3. 多种求解方法组合
        h = np.array(heights)
        methods_results = []
        method_names = []

        # 方法1: 非负最小二乘 (NNLS)
        try:
            ratios1, residual = nnls(G, h)
            if ratios1.sum() > 0:
                ratios1_norm = ratios1 / ratios1.sum()
                methods_results.append(ratios1_norm)
                method_names.append("NNLS")
                print(f"      NNLS结果: {[f'{x:.3f}' for x in ratios1_norm]}, 残差: {residual:.2f}")
        except Exception as e:
            print(f"      NNLS方法失败: {e}")

        # 方法2: Moore-Penrose伪逆
        try:
            G_pinv = np.linalg.pinv(G)
            ratios2 = G_pinv @ h
            ratios2 = np.maximum(ratios2, 0)  # 非负投影
            if ratios2.sum() > 0:
                ratios2_norm = ratios2 / ratios2.sum()
                methods_results.append(ratios2_norm)
                method_names.append("Pinv")
                print(f"      伪逆结果: {[f'{x:.3f}' for x in ratios2_norm]}")
        except Exception as e:
            print(f"      伪逆方法失败: {e}")

        # 方法3: 岭回归 (Ridge Regression)
        try:
            # 手动实现岭回归，避免sklearn依赖
            alpha = 0.01 * np.trace(G.T @ G) / G.shape[1]  # 自适应正则化参数
            I = np.eye(G.shape[1])
            ratios3 = np.linalg.solve(G.T @ G + alpha * I, G.T @ h)
            ratios3 = np.maximum(ratios3, 0)  # 非负投影
            if ratios3.sum() > 0:
                ratios3_norm = ratios3 / ratios3.sum()
                methods_results.append(ratios3_norm)
                method_names.append("Ridge")
                print(f"      岭回归结果: {[f'{x:.3f}' for x in ratios3_norm]} (α={alpha:.4f})")
        except Exception as e:
            print(f"      岭回归方法失败: {e}")

        # 4. 结果融合
        if len(methods_results) == 0:
            return None, "所有求解方法失败"
        
        # 使用中位数融合多种方法的结果
        methods_array = np.array(methods_results)
        final_ratios = np.median(methods_array, axis=0)
        final_ratios = final_ratios / final_ratios.sum()  # 重新归一化
        
        print(f"      成功方法: {method_names}")
        print(f"      中位数融合结果: {[f'{x:.3f}' for x in final_ratios]}")
        
        # 5. 计算拟合质量
        try:
            residual = np.linalg.norm(G @ final_ratios - h)
            relative_error = residual / np.linalg.norm(h)
            print(f"      拟合质量: 残差={residual:.2f}, 相对误差={relative_error:.3f}")
        except:
            pass
        
        return final_ratios, None

    def solve_mixture_ratios(self, sample_data, sample_name):
        """求解混合比例 - 主函数（修改为使用稳健方法）"""
        print(f"\n开始分析样本: {sample_name[:50]}...")
        
        # 1. 解析样本信息
        sample_ids, true_ratios = self.parse_sample_name(sample_name)
        
        if sample_ids is None or true_ratios is None:
            return None, None, "样本名称解析失败"
        
        print(f"  贡献者Sample IDs: {sample_ids}")
        print(f"  真实比例: {true_ratios}")
        
        # 2. 对每个基因座使用稳健求解方法
        all_estimates = []
        successful_markers = []
        failed_info = []
        marker_details = []
        
        # 跳过AM/AMEL标记，只分析常染色体STR
        analysis_markers = [m for m in self.str_markers if m not in ['AMEL', 'AM']]
        
        for marker in analysis_markers:
            # 使用稳健求解方法
            estimate, error = self.solve_single_marker_robust(sample_data, sample_ids, marker)
            
            if estimate is not None:
                all_estimates.append(estimate)
                successful_markers.append(marker)
                marker_details.append({
                    'marker': marker,
                    'estimate': estimate,
                    'error': error
                })
            else:
                failed_info.append(f"{marker}: {error}")
        
        if len(all_estimates) == 0:
            error_msg = f"所有基因座求解失败: {'; '.join(failed_info[:3])}"
            return None, None, error_msg
        
        # 3. 整合多个基因座的结果（二级融合）
        all_estimates = np.array(all_estimates)
        
        # 使用中位数方法进行基因座间融合（稳健）
        final_estimate = np.median(all_estimates, axis=0)
        final_estimate = final_estimate / final_estimate.sum()
        
        # 转换真实比例为归一化形式
        true_ratios_normalized = np.array(true_ratios) / sum(true_ratios)
        
        print(f"\n  === 最终结果 ===")
        print(f"  成功基因座: {len(successful_markers)}/{len(analysis_markers)}")
        print(f"  成功的基因座: {successful_markers}")
        print(f"  最终估算: {[f'{x:.3f}' for x in final_estimate]}")
        print(f"  真实比例: {[f'{x:.3f}' for x in true_ratios_normalized]}")
        
        # 计算简单的准确性指标
        if len(final_estimate) == len(true_ratios_normalized):
            mae = np.mean(np.abs(final_estimate - true_ratios_normalized))
            print(f"  平均绝对误差(MAE): {mae:.4f}")
        
        return final_estimate, true_ratios_normalized, None