import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings('ignore')

# 简化字体设置，避免中文字体问题
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

from data_utils import load_and_process_data
from genotype_solver import GenotypeMixtureSolver
from evaluation import evaluate_accuracy, batch_evaluate, print_evaluation_summary, plot_evaluation_results

def generate_visualizations(mae_scores, contributor_stats, results, output_dir):
    """Generate visualization charts - English version to avoid font issues"""
    
    # 1. MAE Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(mae_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(mae_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(mae_scores):.3f}')
    plt.axvline(np.median(mae_scores), color='green', linestyle='--', 
               label=f'Median: {np.median(mae_scores):.3f}')
    plt.xlabel('Mean Absolute Error (MAE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of MAE for Genotype-based Mixture Ratio Estimation')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'MAE_Distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. MAE Performance by Number of Contributors (bar)
    if contributor_stats:
        categories = sorted(contributor_stats.keys())
        mae_values = [contributor_stats[cat]['mae'] for cat in categories]
        counts = [contributor_stats[cat]['count'] for cat in categories]
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        plt.figure(figsize=(8, 6))
        bars = plt.bar([f'{cat}-person' for cat in categories], mae_values, color=colors[:len(categories)])
        plt.ylabel('Average MAE')
        plt.title('MAE Performance by Number of Contributors')
        plt.grid(alpha=0.3)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.002, f'n={count}', ha='center', va='bottom')
        plt.savefig(os.path.join(output_dir, 'MAE_Performance_By_Contributors.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Sample distribution pie chart
        plt.figure(figsize=(7, 7))
        plt.pie(counts, labels=[f'{cat}-person mixture' for cat in categories], 
               autopct='%1.1f%%', startangle=90, colors=colors[:len(categories)])
        plt.title('Sample Distribution by Number of Contributors')
        plt.savefig(os.path.join(output_dir, 'Sample_Distribution_Pie.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. MAE box plot by category
        mae_data = [contributor_stats[cat]['mae_scores'] for cat in categories]
        plt.figure(figsize=(8, 6))
        plt.boxplot(mae_data, labels=[f'{cat}-person' for cat in categories])
        plt.ylabel('MAE')
        plt.title('MAE Distribution Box Plot by Category')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'MAE_Boxplot_By_Category.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. True vs Estimated scatter plot (first 30 samples)
        plt.figure(figsize=(8, 8))
        sample_results = results[:30]
        colors_scatter = ['red', 'blue', 'green', 'orange', 'purple']
        for i, result in enumerate(sample_results):
            true_ratios = result['true_ratios']
            est_ratios = result['estimated_ratios']
            n_contributors = len(true_ratios)
            for j in range(n_contributors):
                plt.scatter(true_ratios[j], est_ratios[j], color=colors_scatter[j % len(colors_scatter)], alpha=0.6, s=30)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Estimation Line')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('True Ratio')
        plt.ylabel('Estimated Ratio')
        plt.title('True vs Estimated Ratio Scatter Plot')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'True_vs_Estimated_Scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 读取详细评估指标CSV，生成更多可视化图表
    import pandas as pd
    metrics_path = os.path.join(output_dir, 'detailed_evaluation_metrics.csv')
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        # R²分布直方图
        plt.figure(figsize=(8, 6))
        plt.hist(df['r2'].dropna(), bins=20, color='mediumpurple', edgecolor='black', alpha=0.7)
        plt.axvline(df['r2'].mean(), color='red', linestyle='--', label=f'Mean: {df["r2"].mean():.3f}')
        plt.xlabel('R²')
        plt.ylabel('Frequency')
        plt.title('Distribution of R² Values')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'R2_Distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 主贡献者识别准确率柱状图
        acc_by_n = df.groupby('n_contributors')['major_contributor_correct'].mean()
        plt.figure(figsize=(8, 6))
        bars = plt.bar([f'{n}-person' for n in acc_by_n.index], acc_by_n.values, color='#66B2FF')
        plt.ylabel('Major Contributor Identification Accuracy')
        plt.title('Major Contributor Identification Accuracy by Contributors')
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        for i, v in enumerate(acc_by_n.values):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
        plt.savefig(os.path.join(output_dir, 'Major_Contributor_Accuracy_Bar.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 阈值准确率柱状图
        for t, label, color in zip(['threshold_1pct', 'threshold_5pct', 'threshold_10pct'], ['1%', '5%', '10%'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
            acc_by_n = df.groupby('n_contributors')[t].mean()
            plt.figure(figsize=(8, 6))
            bars = plt.bar([f'{n}-person' for n in acc_by_n.index], acc_by_n.values, color=color)
            plt.ylabel(f'Proportion within {label} Error')
            plt.title(f'Proportion of Samples within {label} Error by Contributors')
            plt.ylim(0, 1.05)
            plt.grid(alpha=0.3)
            for i, v in enumerate(acc_by_n.values):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
            plt.savefig(os.path.join(output_dir, f'Threshold_{label}_Accuracy_Bar.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 残差分布直方图
        if 'mae' in df.columns and 'rmse' in df.columns:
            residuals = df['rmse'] - df['mae']
            plt.figure(figsize=(8, 6))
            plt.hist(residuals, bins=20, color='salmon', edgecolor='black', alpha=0.7)
            plt.xlabel('RMSE - MAE')
            plt.ylabel('Frequency')
            plt.title('Distribution of Residuals (RMSE - MAE)')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'Residuals_Distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 熵分布直方图
        if 'entropy_difference' in df.columns:
            plt.figure(figsize=(8, 6))
            plt.hist(df['entropy_difference'].dropna(), bins=20, color='gold', edgecolor='black', alpha=0.7)
            plt.xlabel('Entropy Difference')
            plt.ylabel('Frequency')
            plt.title('Distribution of Entropy Difference')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'Entropy_Difference_Distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

    print("Visualization charts generated successfully")

def main():
    """Main function: Genotype-based mixture ratio solution"""
    np.random.seed(42)
    
    print("="*60)
    print("Forensic DNA Mixture Analysis - Genotype-based Direct Solution")
    print("="*60)
    
    # Create results directory
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Data file paths
    str_data_path = "C:/Users/陈永鸿/Desktop/数学建模/25深圳杯/数据集/附件2：不同混合比例的STR图谱数据.csv"
    genotype_data_path = "C:/Users/陈永鸿/Desktop/数学建模/25深圳杯/数据集/附件3：各个贡献者对应的基因型数据.csv"
    
    # 1. Load STR data
    try:
        str_data = load_and_process_data(str_data_path)
        print(f"Number of samples: {str_data['Sample File'].nunique()}")
    except FileNotFoundError:
        print(f"STR data file not found: {str_data_path}")
        return
    
    # 2. Initialize genotype solver
    solver = GenotypeMixtureSolver()
    if not solver.load_genotype_data(genotype_data_path):
        print("Failed to load genotype data, exiting")
        return
    
    # 3. Analyze all samples
    print(f"\nStarting analysis of {str_data['Sample File'].nunique()} samples...")
    
    results = []
    success_count = 0
    
    for sample_name, sample_group in str_data.groupby('Sample File'):
        # Use genotype method to solve
        estimated_ratios, true_ratios, error = solver.solve_mixture_ratios(
            sample_group, sample_name)
        
        if estimated_ratios is not None and true_ratios is not None:
            # Calculate accuracy
            accuracy = evaluate_accuracy(estimated_ratios, true_ratios)
            
            results.append({
                'sample': sample_name,
                'n_contributors': len(true_ratios),
                'true_ratios': true_ratios,
                'estimated_ratios': estimated_ratios,
                'accuracy': accuracy,
                'success': True
            })
            
            success_count += 1
            print(f"  ✓ MAE: {accuracy['mae']:.3f}")
            
        else:
            print(f"  ✗ Failed: {error}")
            results.append({
                'sample': sample_name,
                'success': False,
                'error': error
            })

    # 4. Detailed Performance Analysis
    print(f"\n" + "="*60)
    print("Detailed Performance Analysis")
    print("="*60)

    successful_results = [r for r in results if r['success']]

    if successful_results:
        try:
            # Use batch evaluation
            evaluation_results = batch_evaluate(successful_results)
            
            # Print detailed summary
            print_evaluation_summary(evaluation_results)
            
            # Generate evaluation visualization
            plot_evaluation_results(evaluation_results, output_dir)
            
            # Save evaluation results to file
            if 'detailed_metrics' in evaluation_results:
                df_metrics = evaluation_results['detailed_metrics']
                df_metrics.to_csv(os.path.join(output_dir, 'detailed_evaluation_metrics.csv'), 
                                index=False, encoding='utf-8-sig')
                print(f"Detailed evaluation metrics saved to {output_dir}/detailed_evaluation_metrics.csv")
        
        except Exception as e:
            print(f"Advanced evaluation failed: {e}")
            print("Continuing with basic evaluation...")
    
    # Basic performance analysis (as backup)
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        mae_scores = [r['accuracy']['mae'] for r in successful_results]
        
        print(f"\nBasic Performance Analysis:")
        print(f"Successfully analyzed samples: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"Overall performance:")
        print(f"  Average MAE: {np.mean(mae_scores):.3f}")
        print(f"  Median MAE: {np.median(mae_scores):.3f}")
        print(f"  Standard deviation: {np.std(mae_scores):.3f}")
        print(f"  Best MAE: {np.min(mae_scores):.3f}")
        print(f"  Worst MAE: {np.max(mae_scores):.3f}")
        print(f"  MAE < 0.1: {np.mean(np.array(mae_scores) < 0.1):.1%}")
        print(f"  MAE < 0.05: {np.mean(np.array(mae_scores) < 0.05):.1%}")
        
        # Group by number of contributors
        print(f"\nGrouped by number of contributors:")
        contributor_stats = {}
        for n in [2, 3, 4, 5]:
            n_results = [r for r in successful_results if r['n_contributors'] == n]
            if n_results:
                n_mae_scores = [r['accuracy']['mae'] for r in n_results]
                avg_mae = np.mean(n_mae_scores)
                contributor_stats[n] = {
                    'count': len(n_results),
                    'mae': avg_mae,
                    'mae_scores': n_mae_scores
                }
                print(f"  {n}-person mixture ({len(n_results)} samples): Average MAE={avg_mae:.3f}")
        
        # 5. Generate visualization results
        generate_visualizations(mae_scores, contributor_stats, successful_results, output_dir)
        
        # 6. Save detailed results
        save_detailed_results(results, output_dir)
        
    else:
        print("No successfully analyzed samples!")
    
    print(f"\nAnalysis completed! Results saved in {os.path.abspath(output_dir)} directory")

def save_detailed_results(results, output_dir):
    """Save detailed results to CSV file"""
    import pandas as pd
    
    # Prepare table data
    table_data = []
    
    for i, result in enumerate(results, 1):
        if result['success']:
            sample_short = result['sample'][:40] + "..." if len(result['sample']) > 40 else result['sample']
            
            table_data.append({
                'No.': i,
                'Sample Name': sample_short,
                'Contributors': result['n_contributors'],
                'True Ratios': '; '.join([f'{r:.3f}' for r in result['true_ratios']]),
                'Estimated Ratios': '; '.join([f'{r:.3f}' for r in result['estimated_ratios']]),
                'MAE': f"{result['accuracy']['mae']:.4f}",
                'RMSE': f"{result['accuracy']['rmse']:.4f}" if 'rmse' in result['accuracy'] else 'N/A',
                'R²': f"{result['accuracy']['r2']:.4f}" if 'r2' in result['accuracy'] else 'N/A',
                'Status': 'Success'
            })
        else:
            table_data.append({
                'No.': i,
                'Sample Name': result['sample'][:40] + "..." if len(result['sample']) > 40 else result['sample'],
                'Status': 'Failed',
                'Error': result.get('error', 'Unknown error')
            })
    
    # Save as CSV
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False, encoding='utf-8-sig')
    
    print("Detailed results table saved")

if __name__ == "__main__":
    main()