import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def evaluate_accuracy(estimated_ratios, true_ratios):
    """
    评估估算比例的准确性
    
    Args:
        estimated_ratios: 估算的混合比例 (numpy array)
        true_ratios: 真实的混合比例 (numpy array)
    
    Returns:
        dict: 包含各种准确性指标的字典
    """
    
    # 确保输入为numpy数组
    estimated = np.array(estimated_ratios)
    true = np.array(true_ratios)
    
    # 基本统计指标
    mae = mean_absolute_error(true, estimated)
    mse = mean_squared_error(true, estimated)
    rmse = np.sqrt(mse)
    
    # 相对误差
    relative_errors = np.abs(true - estimated) / (true + 1e-8)  # 避免除零
    mean_relative_error = np.mean(relative_errors)
    max_relative_error = np.max(relative_errors)
    
    # 相关性指标
    try:
        r2 = r2_score(true, estimated)
        pearson_r, pearson_p = pearsonr(true, estimated)
        spearman_r, spearman_p = spearmanr(true, estimated)
    except:
        r2 = 0
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1
    
    # 自定义准确性指标
    # 1. 最大分量准确性 (主要贡献者识别准确性)
    true_max_idx = np.argmax(true)
    est_max_idx = np.argmax(estimated)
    major_contributor_correct = (true_max_idx == est_max_idx)
    
    # 2. 排序准确性 (Spearman相关系数的绝对值)
    rank_accuracy = abs(spearman_r) if not np.isnan(spearman_r) else 0
    
    # 3. 阈值准确性 (多少比例在阈值范围内)
    threshold_001 = np.mean(np.abs(true - estimated) < 0.01)  # 1%阈值
    threshold_005 = np.mean(np.abs(true - estimated) < 0.05)  # 5%阈值
    threshold_010 = np.mean(np.abs(true - estimated) < 0.10)  # 10%阈值
    
    # 4. 均匀性偏差 (检测是否倾向于估算为均匀分布)
    n_contributors = len(true)
    uniform_dist = np.ones(n_contributors) / n_contributors
    uniformity_bias = mean_absolute_error(estimated, uniform_dist) - mean_absolute_error(true, uniform_dist)
    
    # 5. 混合复杂度适应性
    entropy_true = -np.sum(true * np.log(true + 1e-8))
    entropy_est = -np.sum(estimated * np.log(estimated + 1e-8))
    entropy_difference = abs(entropy_true - entropy_est)
    
    return {
        # 基本误差指标
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mean_relative_error': mean_relative_error,
        'max_relative_error': max_relative_error,
        
        # 相关性指标
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        
        # 法医学特定指标
        'major_contributor_correct': major_contributor_correct,
        'rank_accuracy': rank_accuracy,
        'threshold_1pct': threshold_001,
        'threshold_5pct': threshold_005,
        'threshold_10pct': threshold_010,
        
        # 高级指标
        'uniformity_bias': uniformity_bias,
        'entropy_difference': entropy_difference,
        'n_contributors': n_contributors
    }


# 在现有代码基础上添加缺失的函数

def print_evaluation_summary(evaluation_results):
    """
    打印评估结果摘要
    
    Args:
        evaluation_results: batch_evaluate返回的结果
    """
    
    if 'summary' not in evaluation_results:
        print("No valid evaluation data")
        return
    
    summary = evaluation_results['summary']
    
    print("\n" + "="*80)
    print("Detailed Performance Evaluation Report")
    print("="*80)
    
    # Overall performance
    overall = summary['overall_performance']
    print(f"\n📊 Overall Performance Metrics:")
    print(f"   Total samples: {summary['total_samples']}")
    print(f"   Average MAE: {overall['mean_mae']:.4f}")
    print(f"   Median MAE: {overall['median_mae']:.4f}")
    print(f"   MAE standard deviation: {overall['std_mae']:.4f}")
    print(f"   Average RMSE: {overall['mean_rmse']:.4f}")
    print(f"   Average R²: {overall['mean_r2']:.4f}")
    print(f"   Major contributor identification accuracy: {overall['major_contributor_accuracy']:.1%}")
    print(f"   Ranking accuracy: {overall['rank_accuracy']:.4f}")
    
    # Group by number of contributors
    print(f"\n📈 Analysis by Number of Contributors:")
    for key, stats in summary['by_contributors'].items():
        n_contrib = key.replace('_contributors', '')
        print(f"   {n_contrib}-person mixture ({stats['count']} samples):")
        print(f"     ├─ Average MAE: {stats['mean_mae']:.4f}")
        print(f"     ├─ Median MAE: {stats['median_mae']:.4f}")
        print(f"     ├─ Average R²: {stats['mean_r2']:.4f}")
        print(f"     ├─ Major contributor identification accuracy: {stats['major_contributor_accuracy']:.1%}")
        print(f"     ├─ 5% threshold accuracy: {stats['threshold_5pct']:.1%}")
        print(f"     └─ 10% threshold accuracy: {stats['threshold_10pct']:.1%}")
    
    # Performance grade distribution
    if 'performance_grades' in summary:
        grades = summary['performance_grades']
        percentages = grades['percentages']
        print(f"\n🎯 Performance Grade Distribution:")
        print(f"   Excellent (MAE<5%): {grades['excellent_(<5%)']} samples ({percentages['excellent']:.1f}%)")
        print(f"   Good (MAE 5-10%): {grades['good_(5-10%)']} samples ({percentages['good']:.1f}%)")
        print(f"   Acceptable (MAE 10-20%): {grades['acceptable_(10-20%)']} samples ({percentages['acceptable']:.1f}%)")
        print(f"   Poor (MAE>20%): {grades['poor_(>20%)']} samples ({percentages['poor']:.1f}%)")
    
    # Forensic application assessment
    print(f"\n⚖️ Forensic Application Assessment:")
    excellent_rate = summary['performance_grades']['percentages']['excellent']
    good_rate = summary['performance_grades']['percentages']['good']
    acceptable_rate = excellent_rate + good_rate + summary['performance_grades']['percentages']['acceptable']
    
    if excellent_rate >= 80:
        forensic_grade = "Excellent"
        recommendation = "Highly recommended for forensic evidence identification"
    elif excellent_rate + good_rate >= 80:
        forensic_grade = "Very Good"
        recommendation = "Recommended for forensic evidence identification"
    elif acceptable_rate >= 80:
        forensic_grade = "Good"
        recommendation = "Can be used for forensic evidence identification, suggest combining with other evidence"
    else:
        forensic_grade = "Needs Improvement"
        recommendation = "Method needs further optimization"
    
    print(f"   Forensic grade: {forensic_grade}")
    print(f"   Application recommendation: {recommendation}")
    print(f"   Reliability score: {acceptable_rate:.1f}%")
    
    print("="*80)


def batch_evaluate(results_list):
    """
    批量评估多个样本的结果
    
    Args:
        results_list: 包含多个样本结果的列表，每个元素包含estimated_ratios和true_ratios
    
    Returns:
        dict: 综合评估结果
    """
    
    all_metrics = []
    
    for result in results_list:
        if 'estimated_ratios' in result and 'true_ratios' in result:
            metrics = evaluate_accuracy(result['estimated_ratios'], result['true_ratios'])
            metrics['sample'] = result.get('sample', 'unknown')
            metrics['n_contributors'] = result.get('n_contributors', len(result['true_ratios']))
            all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    # 转换为DataFrame便于分析
    df_metrics = pd.DataFrame(all_metrics)
    
    # 计算总体统计
    summary = {
        'total_samples': len(all_metrics),
        'overall_performance': {
            'mean_mae': df_metrics['mae'].mean(),
            'median_mae': df_metrics['mae'].median(),
            'std_mae': df_metrics['mae'].std(),
            'mean_rmse': df_metrics['rmse'].mean(),
            'mean_r2': df_metrics['r2'].mean(),
            'major_contributor_accuracy': df_metrics['major_contributor_correct'].mean(),
            'rank_accuracy': df_metrics['rank_accuracy'].mean(),
        }
    }
    
    # 按贡献者数量分组分析
    summary['by_contributors'] = {}
    for n_contrib in sorted(df_metrics['n_contributors'].unique()):
        subset = df_metrics[df_metrics['n_contributors'] == n_contrib]
        summary['by_contributors'][f'{n_contrib}_contributors'] = {
            'count': len(subset),
            'mean_mae': subset['mae'].mean(),
            'median_mae': subset['mae'].median(),
            'mean_r2': subset['r2'].mean(),
            'major_contributor_accuracy': subset['major_contributor_correct'].mean(),
            'threshold_5pct': subset['threshold_5pct'].mean(),
            'threshold_10pct': subset['threshold_10pct'].mean(),
        }
    
    # 性能分级
    excellent = len(df_metrics[df_metrics['mae'] < 0.05])
    good = len(df_metrics[(df_metrics['mae'] >= 0.05) & (df_metrics['mae'] < 0.1)])
    acceptable = len(df_metrics[(df_metrics['mae'] >= 0.1) & (df_metrics['mae'] < 0.2)])
    poor = len(df_metrics[df_metrics['mae'] >= 0.2])
    
    summary['performance_grades'] = {
        'excellent_(<5%)': excellent,
        'good_(5-10%)': good,
        'acceptable_(10-20%)': acceptable,
        'poor_(>20%)': poor,
        'percentages': {
            'excellent': excellent / len(all_metrics) * 100,
            'good': good / len(all_metrics) * 100,
            'acceptable': acceptable / len(all_metrics) * 100,
            'poor': poor / len(all_metrics) * 100,
        }
    }
    
    return {
        'summary': summary,
        'detailed_metrics': df_metrics,
        'raw_metrics': all_metrics
    }

# 在plot_evaluation_results函数中修改字体设置部分

def plot_evaluation_results(evaluation_results, output_dir='results'):
    """Generate evaluation result visualizations - English version"""
    
    if 'detailed_metrics' not in evaluation_results:
        print("No valid evaluation data, skipping visualization")
        return
    
    df = evaluation_results['detailed_metrics']
    
    # Set chart style (English only)
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create comprehensive evaluation charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. MAE distribution histogram
    axes[0, 0].hist(df['mae'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['mae'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["mae"].mean():.3f}')
    axes[0, 0].axvline(df['mae'].median(), color='green', linestyle='--', 
                      label=f'Median: {df["mae"].median():.3f}')
    axes[0, 0].set_xlabel('MAE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MAE Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. MAE box plot by number of contributors
    df_grouped = df.groupby('n_contributors')['mae'].apply(list).reset_index()
    mae_by_contributors = [df_grouped[df_grouped['n_contributors']==n]['mae'].iloc[0] 
                          for n in sorted(df['n_contributors'].unique())]
    axes[0, 1].boxplot(mae_by_contributors, 
                      labels=[f'{n}-person' for n in sorted(df['n_contributors'].unique())])
    axes[0, 1].set_xlabel('Number of Contributors')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE Distribution by Number of Contributors')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. R² scatter plot
    scatter = axes[0, 2].scatter(range(len(df)), df['r2'], alpha=0.6, 
                                c=df['n_contributors'], cmap='viridis')
    axes[0, 2].axhline(y=0.8, color='red', linestyle='--', label='Excellent (0.8)')
    axes[0, 2].axhline(y=0.6, color='orange', linestyle='--', label='Good (0.6)')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('R²')
    axes[0, 2].set_title('R² Value Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[0, 2])
    cbar.set_label('Number of Contributors')
    
    # 4. Major contributor identification accuracy
    major_acc_by_n = df.groupby('n_contributors')['major_contributor_correct'].mean()
    bars = axes[1, 0].bar(range(len(major_acc_by_n)), major_acc_by_n.values, 
                         color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'][:len(major_acc_by_n)])
    axes[1, 0].set_xticks(range(len(major_acc_by_n)))
    axes[1, 0].set_xticklabels([f'{n}-person' for n in major_acc_by_n.index])
    axes[1, 0].set_ylabel('Major Contributor Identification Accuracy')
    axes[1, 0].set_title('Major Contributor Identification Accuracy')
    axes[1, 0].grid(alpha=0.3)
    for i, v in enumerate(major_acc_by_n.values):
        axes[1, 0].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    # 5. Threshold accuracy comparison
    thresholds = ['threshold_1pct', 'threshold_5pct', 'threshold_10pct']
    threshold_means = [df[t].mean() for t in thresholds]
    threshold_labels = ['1%', '5%', '10%']
    
    x_pos = np.arange(len(threshold_labels))
    bars = axes[1, 1].bar(x_pos, threshold_means, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(threshold_labels)
    axes[1, 1].set_xlabel('Error Threshold')
    axes[1, 1].set_ylabel('Sample Proportion')
    axes[1, 1].set_title('Accurate Sample Proportion by Error Threshold')
    axes[1, 1].grid(alpha=0.3)
    for i, v in enumerate(threshold_means):
        axes[1, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    # 6. Performance grade pie chart
    if 'performance_grades' in evaluation_results['summary']:
        grades = evaluation_results['summary']['performance_grades']['percentages']
        labels = ['Excellent(<5%)', 'Good(5-10%)', 'Acceptable(10-20%)', 'Poor(>20%)']
        sizes = [grades['excellent'], grades['good'], grades['acceptable'], grades['poor']]
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        
        # Only show non-zero parts
        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
        filtered_labels = [labels[i] for i in non_zero_indices]
        filtered_sizes = [sizes[i] for i in non_zero_indices]
        filtered_colors = [colors[i] for i in non_zero_indices]
        
        axes[1, 2].pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%', 
                      colors=filtered_colors, startangle=90)
        axes[1, 2].set_title('Performance Grade Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Comprehensive_Evaluation_Results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation visualization saved to {output_dir}/Comprehensive_Evaluation_Results.png")