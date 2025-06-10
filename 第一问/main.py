import pandas as pd
import numpy as np
import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mlp import MLP
from visualizer import ModelVisualizer, generate_report
import os

# 加载数据，训练多种模型，评估性能，并生成可视化结果
def get_data():
    file_path = 'C:/Users/陈永鸿/Desktop/数学建模/25深圳杯/数据集/1：不同人数的STR图谱数据.csv'
    df = pd.read_csv(file_path)
    marker_list = df['Marker'].unique().tolist()
    X, y, sample_files, ratios_list = load_data.extract_features(df, marker_list)
    y_unique = sorted(list(set(y)))
    y_map = {v: i for i, v in enumerate(y_unique)}
    y = np.array([y_map[v] for v in y])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, len(y_unique)

# 训练多种模型并评估性能
def train_models():
    print("开始模型训练和评估")
    X_train, X_test, y_train, y_test, num_classes = get_data()
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}, 类别数: {num_classes}")
    
    results = {}
    trained_models = {}
    
    # 逻辑回归
    print("逻辑回归")
    lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    results['Logistic Regression'] = accuracy_score(y_test, lr_pred)
    trained_models['Logistic Regression'] = lr_model
    
    # GBDT
    print("GBDT")
    gbdt_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gbdt_model.fit(X_train, y_train)
    gbdt_pred = gbdt_model.predict(X_test)
    results['GBDT'] = accuracy_score(y_test, gbdt_pred)
    trained_models['GBDT'] = gbdt_model
    
    # XGBoost
    print("XGBoost")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,         
        learning_rate=0.05,         
        max_depth=12,               
        subsample=0.8,              
        colsample_bytree=0.8,       
        reg_alpha=0.1,             
        reg_beta=0.1,               
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results['XGBoost'] = accuracy_score(y_test, xgb_pred)
    trained_models['XGBoost'] = xgb_model
    
    # 朴素贝叶斯
    print("朴素贝叶斯")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    results['GaussianNB'] = accuracy_score(y_test, nb_pred)
    trained_models['GaussianNB'] = nb_model
    
    # MLP
    print("MLP")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = load_data.STRDataset(X_train, y_train)
    test_set = load_data.STRDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8)
    
    mlp_model = MLP(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    mlp_losses = []
    mlp_accs = []
    
    # 训练MLP模型
    for epoch in range(10):
        mlp_model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = mlp_model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * xb.size(0)
            pred_label = pred.argmax(dim=1)
            correct += (pred_label == yb).sum().item()
            total += yb.size(0)
        
        mlp_losses.append(epoch_loss / total)
        mlp_accs.append(correct / total)
    
    mlp_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = mlp_model(xb)
            pred_label = pred.argmax(dim=1)
            correct += (pred_label == yb).sum().item()
            total += yb.size(0)
    
    results['MLP'] = correct / total
    print("模型训练完成")
    
    return results, trained_models, X_test, y_test, num_classes, mlp_losses, mlp_accs

# 生成报告和可视化结果
def main():
    print("STR图谱贡献者人数识别 - 模型分析")
    results, trained_models, X_test, y_test, num_classes, mlp_losses, mlp_accs = train_models()
    
    print("\n模型准确率汇总")
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:20s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("-" * 50)
    
    print("\n生成可视化结果")
    # 指定输出目录
    output_dir = './results'
    visualizer = ModelVisualizer(output_dir=output_dir)
    visualizer.generate_visualizations(
        results=results,
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        num_classes=num_classes,
        mlp_losses=mlp_losses,
        mlp_accs=mlp_accs
    )
    
    generate_report(results, os.path.join(output_dir, "report.txt"))
    
    best_model_name = max(results.items(), key=lambda x: x[1])[0]
    best_accuracy = results[best_model_name]
    
    print(f"\n推荐模型: {best_model_name}")
    print(f"最高准确率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    return results, trained_models

if __name__ == "__main__":
    results, models = main()
