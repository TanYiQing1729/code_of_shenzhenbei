import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as fm
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelVisualizer:
    def __init__(self, output_dir='./results'):
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C']
        self.output_dir = output_dir
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    # 生成准确率排名图    
    def plot_accuracy_ranking(self, results):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(results.keys())
        accuracies = list(results.values())
        
        sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
        models, accuracies = zip(*sorted_data)
        
        bars = ax.bar(models, accuracies, color=self.colors[:len(models)], 
                     alpha=0.8, edgecolor='white', linewidth=2)
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.3f}\n({acc*100:.1f}%)', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            if i == 0:
                bar.set_edgecolor('#FFD700')
                bar.set_linewidth(3)
        
        avg_acc = np.mean(accuracies)
        ax.axhline(y=avg_acc, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(len(models)-1, avg_acc + 0.02, f'平均值: {avg_acc:.3f}', 
               fontsize=10, color='red', fontweight='bold')
        
        ax.set_title('模型准确率排名对比', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('准确率', fontsize=14, fontweight='bold')
        ax.set_xlabel('模型', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(accuracies) * 1.15)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 生成MLP训练曲线图    
    def plot_mlp_training(self, mlp_losses, mlp_accs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(mlp_losses) + 1)
        
        ax1.plot(epochs, mlp_losses, 'o-', color='#E74C3C', linewidth=3, markersize=8, alpha=0.8)
        ax1.set_title('MLP训练损失变化', fontsize=14, fontweight='bold')
        ax1.set_xlabel('训练轮次', fontsize=12)
        ax1.set_ylabel('损失值', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        min_loss_idx = np.argmin(mlp_losses)
        ax1.annotate(f'最低损失: {mlp_losses[min_loss_idx]:.4f}', 
                    xy=(min_loss_idx+1, mlp_losses[min_loss_idx]),
                    xytext=(min_loss_idx+1, mlp_losses[min_loss_idx]+0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, ha='center')
        
        ax2.plot(epochs, mlp_accs, 's-', color='#27AE60', linewidth=3, markersize=8, alpha=0.8)
        ax2.set_title('MLP训练准确率变化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('训练轮次', fontsize=12)
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        max_acc_idx = np.argmax(mlp_accs)
        ax2.annotate(f'最高准确率: {mlp_accs[max_acc_idx]:.4f}', 
                    xy=(max_acc_idx+1, mlp_accs[max_acc_idx]),
                    xytext=(max_acc_idx+1, mlp_accs[max_acc_idx]-0.05),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mlp_training.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 生成混淆矩阵对比图    
    def plot_confusion_matrices(self, trained_models, X_test, y_test, num_classes):
        top_models = ['GBDT', 'XGBoost', 'Logistic Regression']
        available_models = [model for model in top_models if model in trained_models]
        
        if not available_models:
            available_models = list(trained_models.keys())[:3]
        
        fig, axes = plt.subplots(1, len(available_models), figsize=(5*len(available_models), 4))
        if len(available_models) == 1:
            axes = [axes]
        
        class_labels = [f'{i+1}人' for i in range(num_classes)]
        
        for idx, model_name in enumerate(available_models):
            model = trained_models[model_name]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[idx].set_title(f'{model_name}\n准确率: {np.trace(cm)/np.sum(cm):.3f}', 
                              fontsize=12, fontweight='bold')
            
            axes[idx].set_xticks(range(num_classes))
            axes[idx].set_yticks(range(num_classes))
            axes[idx].set_xticklabels(class_labels)
            axes[idx].set_yticklabels(class_labels)
            axes[idx].set_xlabel('预测标签', fontsize=10)
            axes[idx].set_ylabel('真实标签', fontsize=10)
            
            for i in range(num_classes):
                for j in range(num_classes):
                    text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    axes[idx].text(j, i, f'{cm[i, j]}', ha='center', va='center',
                                 color=text_color, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 生成所有可视化图表并保存    
    def generate_visualizations(self, results, trained_models, X_test, y_test, 
                              num_classes, mlp_losses, mlp_accs):
        print(f"图片将保存到: {os.path.abspath(self.output_dir)}")
        print("1. 生成准确率排名图...")
        self.plot_accuracy_ranking(results)
        
        print("2. 生成MLP训练曲线...")
        self.plot_mlp_training(mlp_losses, mlp_accs)
        
        print("3. 生成混淆矩阵对比图...")
        self.plot_confusion_matrices(trained_models, X_test, y_test, num_classes)
        
        print("可视化生成完成")

# 生成模型评估报告
def generate_report(results, output_path='./results/report.txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("STR图谱贡献者人数识别 - 模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("模型准确率排名:\n")
        for i, (model_name, accuracy) in enumerate(sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
            f.write(f"{i}. {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        f.write(f"\n平均准确率: {np.mean(list(results.values())):.4f}\n")
        f.write(f"最高准确率: {max(results.values()):.4f}\n")
        f.write(f"最低准确率: {min(results.values()):.4f}\n")