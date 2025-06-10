import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Concatenate, Input, Flatten
from tensorflow.keras.models import Model
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置TensorFlow GPU内存自适应增长
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 解析样本名称，提取贡献者和比例
def parse_sample_name(name):
    match = re.search(r'RD14-0003-(\d+)_?(\d+)?_?(\d+)?_?(\d+)?[;-]', name)
    if not match:
        return None, None
    contributors = [int(x) for x in match.groups() if x is not None]
    ratio_match = re.search(r'(\d+);(\d+);(\d+);(\d+)', name)
    if ratio_match:
        ratios = [int(x) for x in ratio_match.groups()]
    else:
        ratios = [1] * len(contributors)
    return contributors, ratios

# 加载并预处理数据
def load_and_preprocess(file_path, genotype_df):
    df = pd.read_csv(file_path, skipinitialspace=True)
    melted_dfs = []
    # 处理样本名称和提取贡献者信息
    for idx, row in df.iterrows():
        sample_name = row['Sample File']
        marker = row['Marker']
        contributors, ratios = parse_sample_name(sample_name)
        # 如果没有找到贡献者或比例信息，则跳过该样本
        if contributors is None:
            continue
        sizes, heights = [], []
        # 提取大小和高度信息
        for i in range(1, 101):
            size_col = f'Size {i}'
            height_col = f'Height {i}'
            # 检查列是否存在且非空
            if size_col in row and pd.notna(row[size_col]) and row[size_col] != 'OL':
                try:
                    size_val = float(row[size_col])
                    height_val = float(row[height_col])
                    sizes.append(size_val)
                    heights.append(height_val)
                except (ValueError, TypeError):
                    continue
        true_genotypes = []
        # 处理基因型数据
        for contrib in contributors:
            contrib_str = str(contrib)
            # 检查基因型数据中是否存在该贡献者
            if marker not in genotype_df.columns:
                continue
            genotype_row = genotype_df[genotype_df['Sample ID'].astype(str) == contrib_str]
            # 如果没有找到对应的基因型数据，则跳过
            if genotype_row.empty:
                continue
            genotype_str = genotype_row[marker].values[0]
            # 如果基因型数据为空或NaN，则跳过
            if pd.isna(genotype_str):
                continue
            try:
                alleles = [float(x) for x in str(genotype_str).split(',') if x.strip() != '']
                true_genotypes.extend(sorted(alleles))
            except Exception:
                continue
        melted_dfs.append({
            'sample': sample_name,
            'marker': marker,
            'contributors': contributors,
            'ratios': ratios,
            'num_contributors': len(contributors),
            'sizes': sizes,
            'heights': heights,
            'true_genotypes': true_genotypes
        })
    return pd.DataFrame(melted_dfs)

# 创建分箱特征
def create_binned_features(sizes, heights, min_size, max_size, num_bins=100):
    bins = np.linspace(min_size, max_size, num_bins)
    bin_heights = np.zeros(num_bins)
    for s, h in zip(sizes, heights):
        bin_idx = np.digitize(s, bins) - 1
        if 0 <= bin_idx < num_bins:
            bin_heights[bin_idx] += h
    if np.max(bin_heights) > 0:
        bin_heights /= np.max(bin_heights)
    return bin_heights, bins

# 准备特征和标签
def prepare_features(df, num_bins=100):
    features_list, labels_list, groups_list, marker_stats, marker_list = [], [], [], {}, []
    # 统计每个marker的大小范围
    for marker in df['marker'].unique():
        marker_data = df[df['marker'] == marker]
        all_sizes = np.concatenate(marker_data['sizes'].values)
        marker_stats[marker] = {
            'min': np.min(all_sizes) if len(all_sizes) > 0 else 0,
            'max': np.max(all_sizes) if len(all_sizes) > 0 else 1
        }
    # 遍历每一行数据，提取特征和标签
    for _, row in df.iterrows():
        sizes = row['sizes']
        heights = row['heights']
        marker = row['marker']
        num_contrib = row['num_contributors']
        true_genotypes = row['true_genotypes']
        mmin = marker_stats[marker]['min']
        mmax = marker_stats[marker]['max']
        bin_features, _ = create_binned_features(sizes, heights, mmin, mmax, num_bins)
        # 统计特征增强
        stat_features = [
            np.mean(heights) if len(heights) else 0,
            np.std(heights) if len(heights) else 0,
            np.max(heights) if len(heights) else 0,
            np.min(heights) if len(heights) else 0,
            len(sizes),
            np.mean(sizes) if len(sizes) else 0,
            np.std(sizes) if len(sizes) else 0,
            np.max(sizes) if len(sizes) else 0,
            np.min(sizes) if len(sizes) else 0
        ]
        feat = np.concatenate([bin_features, stat_features])
        features_list.append(feat)
        label = np.array(true_genotypes)
        if len(label) < 4:
            label = np.pad(label, (0, 4 - len(label)))
        labels_list.append(label)
        groups_list.append(row['sample'])
        marker_list.append(marker)
    # 允许多人混合样本，自动适配不同等位基因数
    # 统计所有样本真实等位基因数的最大值
    max_allele_count = max(len(lab) for lab in labels_list)
    # 统一标签长度，右侧补0
    features_list_new, labels_list_new, groups_list_new, marker_list_new = [], [], [], []
    for feat, lab, grp, mkr in zip(features_list, labels_list, groups_list, marker_list):
        if np.count_nonzero(lab) >= 2:  # 至少2个等位基因
            pad_lab = np.pad(lab, (0, max_allele_count - len(lab)))
            features_list_new.append(feat)
            labels_list_new.append(pad_lab)
            groups_list_new.append(grp)
            marker_list_new.append(mkr)
    features_list, labels_list, groups_list, marker_list = features_list_new, labels_list_new, groups_list_new, marker_list_new
    return np.array(features_list), np.array(labels_list), groups_list, marker_list

# 创建MLP模型
def create_model(input_dim, output_dim=4, use_huber=False):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(output_dim, activation='linear')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=huber_loss if use_huber else masked_mse,
        metrics=['mae']
    )
    return model

# 创建CNN模型
def create_cnn_model(input_dim, output_dim=4, num_bins=100, stat_dim=9, use_huber=False):
    # 输入：分箱特征+统计特征
    input_bins = Input(shape=(num_bins, 1), name='bin_input')
    input_stat = Input(shape=(stat_dim,), name='stat_input')
    x = Conv1D(32, 5, activation='relu', padding='same')(input_bins)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = GlobalMaxPooling1D()(x)
    x = Concatenate()([x, input_stat])
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(output_dim, activation='linear')(x)
    model = Model(inputs=[input_bins, input_stat], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=huber_loss if use_huber else masked_mse,
        metrics=['mae']
    )
    return model

# 定义损失函数：只对非0标签计算损失
def masked_mse(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    squared_error = tf.square(y_true - y_pred)
    masked_error = squared_error * mask
    return tf.reduce_sum(masked_error) / tf.maximum(tf.reduce_sum(mask), 1)

# 定义Huber损失函数：对非0标签计算
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = 0.5 * tf.square(quadratic) + delta * linear
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / tf.maximum(tf.reduce_sum(mask), 1)

# 评估预测结果
def evaluate_predictions(y_true, y_pred, threshold=1.0):

    allele_correct = 0
    genotype_correct = 0
    total_alleles = 0
    total_genotypes = 0
    for i in range(len(y_true)):
        true_vals = y_true[i]
        pred_vals = y_pred[i]
        # 只统计非零等位基因
        true_nonzero = true_vals[true_vals > 0]
        pred_nonzero = pred_vals[:len(true_nonzero)]
        diff = np.abs(true_nonzero - pred_nonzero)
        allele_correct += np.sum(diff <= threshold)
        total_alleles += len(true_nonzero)
        # 按每2个等位基因为一组配对
        if len(true_nonzero) % 2 == 0:
            n = len(true_nonzero) // 2
            true_sorted = np.sort(true_nonzero.reshape(n, 2), axis=1)
            pred_sorted = np.sort(pred_nonzero.reshape(n, 2), axis=1)
            for j in range(n):
                if np.all(np.abs(true_sorted[j] - pred_sorted[j]) <= threshold):
                    genotype_correct += 1
            total_genotypes += n
    allele_acc = allele_correct / total_alleles if total_alleles > 0 else 0
    genotype_acc = genotype_correct / total_genotypes if total_genotypes > 0 else 0
    return allele_acc, genotype_acc

def main():
    # 读取基因型数据
    genotype_df = pd.read_csv('C:/Users/陈永鸿/Desktop/数学建模/25深圳杯/数据集/附件3：各个贡献者对应的基因型数据.csv')
    genotype_df.columns = [col.strip() for col in genotype_df.columns]
    # 读取STR图谱数据
    df1 = load_and_preprocess('C:/Users/陈永鸿/Desktop/数学建模/25深圳杯/数据集/1：不同人数的STR图谱数据.csv', genotype_df)
    df2 = load_and_preprocess('C:/Users/陈永鸿/Desktop/数学建模/25深圳杯/数据集/附件2：不同混合比例的STR图谱数据.csv', genotype_df)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    X, y, groups, marker_list = prepare_features(combined_df, num_bins=100)
    # 标签归一化
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y)
    # 输出当前最大等位基因数
    print(f"当前最大等位基因数: {y.shape[1]}")
    # 分组划分
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = [groups[i] for i in train_idx]
        groups_test = [groups[i] for i in test_idx]
        marker_train = [marker_list[i] for i in train_idx]
        marker_test = [marker_list[i] for i in test_idx]
    # 特征归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # MLP模型
    model = create_model(input_dim=X_train.shape[1], output_dim=y.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    # 随机森林集成
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    # XGBoost集成
    xgb = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, tree_method='auto')
    xgb.fit(X_train, y_train)
    # 预测与反归一化
    y_pred_mlp = model.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)
    # 集成：加权平均（可调整权重）
    y_pred = 0.5 * y_pred_mlp + 0.3 * y_pred_rf + 0.2 * y_pred_xgb
    y_test_inv = y_scaler.inverse_transform(y_test)
    y_pred_inv = y_scaler.inverse_transform(y_pred)
    # 输出前20组
    print("预测结果与真实结果（前20组）：")
    for i in range(min(20, len(y_test_inv))):
        print(f"样本 {groups_test[i]}")
        print("真实基因型:", np.round(y_test_inv[i], 2))
        print("预测基因型:", np.round(y_pred_inv[i], 2))
        print("-" * 40)
    # 评估
    allele_acc, genotype_acc = evaluate_predictions(y_test_inv, y_pred_inv, threshold=1.0)
    print(f"等位基因级别准确率: {allele_acc:.4f}")
    print(f"基因型级别准确率: {genotype_acc:.4f}")
    # 直接输出到标准输出，便于查看
    print(f"最终准确率输出：等位基因级别准确率={allele_acc:.4f}，基因型级别准确率={genotype_acc:.4f}")

    # 分marker准确率统计
    marker_acc = {}
    for marker in set(marker_test):
        idxs = [i for i, m in enumerate(marker_test) if m == marker]
        if not idxs:
            continue
        y_true_marker = y_test_inv[idxs]
        y_pred_marker = y_pred_inv[idxs]
        allele_acc_m, genotype_acc_m = evaluate_predictions(y_true_marker, y_pred_marker, threshold=1.0)
        marker_acc[marker] = (allele_acc_m, genotype_acc_m)
    print("\n分marker准确率:")
    for marker, (aacc, gacc) in marker_acc.items():
        print(f"{marker}: 等位基因准确率={aacc:.3f}, 基因型准确率={gacc:.3f}")

    # 分样本准确率统计
    print("\n分样本准确率:")
    for i in range(len(y_test_inv)):
        true_vals = y_test_inv[i]
        pred_vals = y_pred_inv[i]
        diff = np.abs(true_vals - pred_vals)
        allele_acc_sample = np.sum(diff <= 1.0) / 4
        print(f"样本 {groups_test[i]}: 等位基因准确率={allele_acc_sample:.2f}")
    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='训练MAE')
    plt.plot(history.history['val_mae'], label='验证MAE')
    plt.title('平均绝对误差')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    # 保存结果
    result_rows = []
    for i in range(len(y_test_inv)):
        result_rows.append({
            '样本名': groups_test[i],
            '真实基因型': ','.join(map(str, np.round(y_test_inv[i], 2))),
            '预测基因型': ','.join(map(str, np.round(y_pred_inv[i], 2)))
        })
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
    print("预测结果已保存到 prediction_results.csv")
    mse = np.mean((y_test_inv - y_pred_inv) ** 2)
    mae = np.mean(np.abs(y_test_inv - y_pred_inv))
    with open('mse.txt', 'w', encoding='utf-8') as f:
        f.write(f"均方误差（MSE）: {mse:.4f}\n")
    with open('mae.txt', 'w', encoding='utf-8') as f:
        f.write(f"平均绝对误差（MAE）: {mae:.4f}\n")
    with open('accuracy.txt', 'w', encoding='utf-8') as f:
        f.write(f"等位基因级别准确率: {allele_acc:.4f}\n")
        f.write(f"基因型级别准确率: {genotype_acc:.4f}\n")
    print("MSE/MAE/准确率已保存到文件")
    model.save('str_genotype_predictor.h5')

    # 特征重要性分析（用RF和XGB）
    importances_rf = rf.feature_importances_
    importances_xgb = xgb.feature_importances_
    avg_importance = (importances_rf + importances_xgb) / 2
    # 选取重要性排名前80%的特征
    threshold = np.percentile(avg_importance, 20)  # 剔除最不重要的20%
    selected_idx = np.where(avg_importance > threshold)[0]
    print(f"自动筛选后保留特征数: {len(selected_idx)}/{len(avg_importance)}")
    # 重新训练模型（仅用重要特征）
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]
    # 重新训练MLP
    model_sel = create_model(input_dim=X_train_sel.shape[1], output_dim=y.shape[1])
    history_sel = model_sel.fit(
        X_train_sel, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    # 重新训练RF和XGB
    rf_sel = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train_sel, y_train)
    xgb_sel = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, tree_method='auto')
    xgb_sel.fit(X_train_sel, y_train)
    # 预测与集成
    y_pred_mlp_sel = model_sel.predict(X_test_sel)
    y_pred_rf_sel = rf_sel.predict(X_test_sel)
    y_pred_xgb_sel = xgb_sel.predict(X_test_sel)
    y_pred_sel = 0.5 * y_pred_mlp_sel + 0.3 * y_pred_rf_sel + 0.2 * y_pred_xgb_sel
    y_pred_inv_sel = y_scaler.inverse_transform(y_pred_sel)
    # 输出新准确率
    allele_acc_sel, genotype_acc_sel = evaluate_predictions(y_test_inv, y_pred_inv_sel, threshold=1.0)
    print(f"自动特征筛选后准确率：等位基因级别={allele_acc_sel:.4f}，基因型级别={genotype_acc_sel:.4f}")

    # MLP模型（Huber loss）
    model_huber = create_model(input_dim=X_train.shape[1], output_dim=y.shape[1], use_huber=True)
    history_huber = model_huber.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    y_pred_mlp_huber = model_huber.predict(X_test)
    y_pred_huber = 0.5 * y_pred_mlp_huber + 0.3 * y_pred_rf + 0.2 * y_pred_xgb
    y_pred_inv_huber = y_scaler.inverse_transform(y_pred_huber)
    allele_acc_huber, genotype_acc_huber = evaluate_predictions(y_test_inv, y_pred_inv_huber, threshold=1.0)
    print(f"Huber loss集成后准确率：等位基因级别={allele_acc_huber:.4f}，基因型级别={genotype_acc_huber:.4f}")

    # 分marker单独建模
    print("\n===== 分marker单独建模结果 =====")
    unique_markers = sorted(set(marker_list))
    marker_accs = []
    for marker in unique_markers:
        idxs = [i for i, m in enumerate(marker_list) if m == marker]
        if len(idxs) < 10:
            continue  # 样本太少不建模
        X_marker = X[idxs]
        y_marker = y[idxs]
        groups_marker = [groups[i] for i in idxs]
        # 分组划分
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_idx, test_idx in gss.split(X_marker, y_marker, groups=groups_marker):
            X_train_m, X_test_m = X_marker[train_idx], X_marker[test_idx]
            y_train_m, y_test_m = y_marker[train_idx], y_marker[test_idx]
        scaler_m = StandardScaler()
        X_train_m = scaler_m.fit_transform(X_train_m)
        X_test_m = scaler_m.transform(X_test_m)
        # 训练模型
        model_m = create_model(input_dim=X_train_m.shape[1], output_dim=y.shape[1])
        early_stop_m = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        model_m.fit(X_train_m, y_train_m, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stop_m], verbose=0)
        rf_m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_m.fit(X_train_m, y_train_m)
        xgb_m = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, tree_method='auto')
        xgb_m.fit(X_train_m, y_train_m)
        # 预测与反归一化
        y_pred_mlp_m = model_m.predict(X_test_m)
        y_pred_rf_m = rf_m.predict(X_test_m)
        y_pred_xgb_m = xgb_m.predict(X_test_m)
        y_pred_m = 0.5 * y_pred_mlp_m + 0.3 * y_pred_rf_m + 0.2 * y_pred_xgb_m
        y_scaler_m = MinMaxScaler()
        y_scaler_m.fit(y_marker)
        y_test_inv_m = y_scaler_m.inverse_transform(y_test_m)
        y_pred_inv_m = y_scaler_m.inverse_transform(y_pred_m)
        allele_acc_m, genotype_acc_m = evaluate_predictions(y_test_inv_m, y_pred_inv_m, threshold=1.0)
        marker_accs.append((allele_acc_m, genotype_acc_m, marker))
        print(f"{marker}: 等位基因准确率={allele_acc_m:.4f}, 基因型准确率={genotype_acc_m:.4f}, 测试样本数={len(y_test_m)}")
    if marker_accs:
        mean_allele = np.mean([a for a, g, m in marker_accs])
        mean_geno = np.mean([g for a, g, m in marker_accs])
        print(f"\n分marker单独建模平均准确率：等位基因级别={mean_allele:.4f}，基因型级别={mean_geno:.4f}")
    else:
        print("分marker样本数过少，未能建模")

    # 结果可视化：预测与真实的散点图
    plt.figure(figsize=(8, 6))
    y_true_flat = y_test_inv.flatten()
    y_pred_flat = y_pred_inv.flatten()
    mask = y_true_flat > 0  # 只画真实等位基因非零的点
    plt.scatter(y_true_flat[mask], y_pred_flat[mask], alpha=0.5, s=20, c='b', label='预测值')
    plt.plot([y_true_flat[mask].min(), y_true_flat[mask].max()], [y_true_flat[mask].min(), y_true_flat[mask].max()], 'r--', label='理想预测')
    plt.xlabel('真实等位基因')
    plt.ylabel('预测等位基因')
    plt.title('等位基因预测结果散点图')
    plt.legend()
    plt.tight_layout()
    plt.savefig('scatter_true_vs_pred.png')
    plt.show()
    print('预测与真实等位基因散点图已保存为 scatter_true_vs_pred.png')

    # 结果可视化：分marker准确率柱状图
    if marker_acc:
        plt.figure(figsize=(12, 5))
        markers = list(marker_acc.keys())
        allele_accs = [marker_acc[m][0] for m in markers]
        geno_accs = [marker_acc[m][1] for m in markers]
        x = np.arange(len(markers))
        plt.bar(x - 0.2, allele_accs, width=0.4, label='等位基因准确率')
        plt.bar(x + 0.2, geno_accs, width=0.4, label='基因型准确率')
        plt.xticks(x, markers, rotation=45)
        plt.ylabel('准确率')
        plt.title('分marker准确率')
        plt.legend()
        plt.tight_layout()
        plt.savefig('marker_accuracy_bar.png')
        plt.show()
        print('分marker准确率柱状图已保存为 marker_accuracy_bar.png')

    # 结果可视化：分样本准确率直方图
    sample_allele_accs = []
    for i in range(len(y_test_inv)):
        true_vals = y_test_inv[i]
        pred_vals = y_pred_inv[i]
        true_nonzero = true_vals[true_vals > 0]
        pred_nonzero = pred_vals[:len(true_nonzero)]
        diff = np.abs(true_nonzero - pred_nonzero)
        allele_acc_sample = np.sum(diff <= 1.0) / len(true_nonzero)
        sample_allele_accs.append(allele_acc_sample)
    plt.figure(figsize=(8, 5))
    plt.hist(sample_allele_accs, bins=20, color='skyblue', edgecolor='k')
    plt.xlabel('单样本等位基因准确率')
    plt.ylabel('样本数')
    plt.title('分样本等位基因准确率分布')
    plt.tight_layout()
    plt.savefig('sample_accuracy_hist.png')
    plt.show()
    print('分样本等位基因准确率分布图已保存为 sample_accuracy_hist.png')

    # 结果可视化：预测误差分布
    error_flat = np.abs(y_true_flat[mask] - y_pred_flat[mask])
    plt.figure(figsize=(8, 5))
    plt.hist(error_flat, bins=30, color='salmon', edgecolor='k')
    plt.xlabel('预测误差（绝对值）')
    plt.ylabel('等位基因数')
    plt.title('预测误差分布')
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.show()
    print('预测误差分布图已保存为 error_distribution.png')

    # 分marker单独建模结果可视化
    if marker_accs:
        markers = [m for a, g, m in marker_accs]
        allele_accs = [a for a, g, m in marker_accs]
        geno_accs = [g for a, g, m in marker_accs]
        x = np.arange(len(markers))
        plt.figure(figsize=(14, 6))
        bar1 = plt.bar(x - 0.2, allele_accs, width=0.4, label='等位基因准确率', color='skyblue')
        bar2 = plt.bar(x + 0.2, geno_accs, width=0.4, label='基因型准确率', color='salmon')
        plt.axhline(mean_allele, color='blue', linestyle='--', label=f'等位基因均值={mean_allele:.3f}')
        plt.axhline(mean_geno, color='red', linestyle='--', label=f'基因型均值={mean_geno:.3f}')
        plt.xticks(x, markers, rotation=45)
        plt.ylabel('准确率')
        plt.ylim(0, 1.05)
        plt.title('分marker单独建模准确率')
        plt.legend()
        plt.tight_layout()
        plt.savefig('marker_individual_model_accuracy.png')
        plt.show()
        print('分marker单独建模准确率图已保存为 marker_individual_model_accuracy.png')

    # 分marker单独建模训练过程可视化
    if marker_accs:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for marker in [m for a, g, m in marker_accs]:
            idxs = [i for i, mkr in enumerate(marker_list) if mkr == marker]
            if len(idxs) < 10:
                continue
            X_marker = X[idxs]
            y_marker = y[idxs]
            groups_marker = [groups[i] for i in idxs]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            for train_idx, test_idx in gss.split(X_marker, y_marker, groups=groups_marker):
                X_train_m, X_test_m = X_marker[train_idx], X_marker[test_idx]
                y_train_m, y_test_m = y_marker[train_idx], y_marker[test_idx]
            scaler_m = StandardScaler()
            X_train_m = scaler_m.fit_transform(X_train_m)
            X_test_m = scaler_m.transform(X_test_m)
            model_m = create_model(input_dim=X_train_m.shape[1], output_dim=y.shape[1])
            early_stop_m = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
            history_m = model_m.fit(X_train_m, y_train_m, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stop_m], verbose=0)
            axes[0].plot(history_m.history['loss'], label=f'{marker}')
            axes[1].plot(history_m.history['val_loss'], label=f'{marker}')
        axes[0].set_title('分marker单独建模训练损失')
        axes[0].set_ylabel('训练损失')
        axes[1].set_title('分marker单独建模验证损失')
        axes[1].set_ylabel('验证损失')
        axes[1].set_xlabel('Epoch')
        axes[0].legend(fontsize=8, ncol=3)
        axes[1].legend(fontsize=8, ncol=3)
        plt.tight_layout()
        plt.savefig('marker_individual_training_history.png')
        plt.show()
        print('分marker单独建模训练过程可视化已保存为 marker_individual_training_history.png')

    # 分marker单独建模训练过程正确率可视化
    if marker_accs:
        fig_acc, axes_acc = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for marker in [m for a, g, m in marker_accs]:
            idxs = [i for i, mkr in enumerate(marker_list) if mkr == marker]
            if len(idxs) < 10:
                continue
            X_marker = X[idxs]
            y_marker = y[idxs]
            groups_marker = [groups[i] for i in idxs]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            for train_idx, test_idx in gss.split(X_marker, y_marker, groups=groups_marker):
                X_train_m, X_test_m = X_marker[train_idx], X_marker[test_idx]
                y_train_m, y_test_m = y_marker[train_idx], y_marker[test_idx]
            scaler_m = StandardScaler()
            X_train_m = scaler_m.fit_transform(X_train_m)
            X_test_m = scaler_m.transform(X_test_m)
            model_m = create_model(input_dim=X_train_m.shape[1], output_dim=y.shape[1])
            early_stop_m = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
            history_m = model_m.fit(X_train_m, y_train_m, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stop_m], verbose=0)
            # 计算每个epoch的训练集和验证集准确率
            train_accs = []
            val_accs = []
            y_train_pred = model_m.predict(X_train_m)
            y_train_inv = MinMaxScaler().fit(y_marker).inverse_transform(y_train_m)
            y_train_pred_inv = MinMaxScaler().fit(y_marker).inverse_transform(y_train_pred)
            train_acc, _ = evaluate_predictions(y_train_inv, y_train_pred_inv, threshold=1.0)
            train_accs.append(train_acc)
            # 验证集
            y_val_pred = model_m.predict(X_test_m)
            y_val_inv = MinMaxScaler().fit(y_marker).inverse_transform(y_test_m)
            y_val_pred_inv = MinMaxScaler().fit(y_marker).inverse_transform(y_val_pred)
            val_acc, _ = evaluate_predictions(y_val_inv, y_val_pred_inv, threshold=1.0)
            val_accs.append(val_acc)
            axes_acc[0].plot([train_acc]*len(history_m.history['loss']), label=f'{marker}')
            axes_acc[1].plot([val_acc]*len(history_m.history['loss']), label=f'{marker}')
        axes_acc[0].set_title('分marker单独建模训练集等位基因准确率')
        axes_acc[0].set_ylabel('训练集准确率')
        axes_acc[1].set_title('分marker单独建模验证集等位基因准确率')
        axes_acc[1].set_ylabel('验证集准确率')
        axes_acc[1].set_xlabel('Epoch')
        axes_acc[0].legend(fontsize=8, ncol=3)
        axes_acc[1].legend(fontsize=8, ncol=3)
        plt.tight_layout()
        plt.savefig('marker_individual_training_accuracy.png')
        plt.show()
        print('分marker单独建模训练过程准确率可视化已保存为 marker_individual_training_accuracy.png')

    # 分marker单独建模训练过程动态准确率可视化
    if marker_accs:
        fig_acc, axes_acc = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for marker in [m for a, g, m in marker_accs]:
            idxs = [i for i, mkr in enumerate(marker_list) if mkr == marker]
            if len(idxs) < 10:
                continue
            X_marker = X[idxs]
            y_marker = y[idxs]
            groups_marker = [groups[i] for i in idxs]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            for train_idx, test_idx in gss.split(X_marker, y_marker, groups=groups_marker):
                X_train_m, X_test_m = X_marker[train_idx], X_marker[test_idx]
                y_train_m, y_test_m = y_marker[train_idx], y_marker[test_idx]
            scaler_m = StandardScaler()
            X_train_m = scaler_m.fit_transform(X_train_m)
            X_test_m = scaler_m.transform(X_test_m)
            model_m = create_model(input_dim=X_train_m.shape[1], output_dim=y.shape[1])
            early_stop_m = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
            # 动态准确率记录
            train_accs = []
            val_accs = []
            y_scaler_m = MinMaxScaler().fit(y_marker)
            for epoch in range(1, 201):
                model_m.fit(X_train_m, y_train_m, epochs=1, batch_size=32, validation_split=0.1, callbacks=[], verbose=0)
                # 训练集准确率
                y_train_pred = model_m.predict(X_train_m, verbose=0)
                y_train_inv = y_scaler_m.inverse_transform(y_train_m)
                y_train_pred_inv = y_scaler_m.inverse_transform(y_train_pred)
                train_acc, _ = evaluate_predictions(y_train_inv, y_train_pred_inv, threshold=1.0)
                train_accs.append(train_acc)
                # 验证集准确率
                y_val_pred = model_m.predict(X_test_m, verbose=0)
                y_val_inv = y_scaler_m.inverse_transform(y_test_m)
                y_val_pred_inv = y_scaler_m.inverse_transform(y_val_pred)
                val_acc, _ = evaluate_predictions(y_val_inv, y_val_pred_inv, threshold=1.0)
                val_accs.append(val_acc)
            axes_acc[0].plot(train_accs, label=f'{marker}')
            axes_acc[1].plot(val_accs, label=f'{marker}')
        axes_acc[0].set_title('分marker单独建模训练集等位基因准确率（动态）')
        axes_acc[0].set_ylabel('训练集准确率')
        axes_acc[1].set_title('分marker单独建模验证集等位基因准确率（动态）')
        axes_acc[1].set_ylabel('验证集准确率')
        axes_acc[1].set_xlabel('Epoch')
        axes_acc[0].legend(fontsize=8, ncol=3)
        axes_acc[1].legend(fontsize=8, ncol=3)
        plt.tight_layout()
        plt.savefig('marker_individual_training_accuracy_dynamic.png')
        plt.show()
        print('分marker单独建模训练过程动态准确率可视化已保存为 marker_individual_training_accuracy_dynamic.png')

    # 统计标签补零比例，分析损失低的原因
    zero_ratios = [np.sum(lab == 0) / len(lab) for lab in y]
    plt.figure(figsize=(8, 4))
    plt.hist(zero_ratios, bins=20, color='orange', edgecolor='k')
    plt.xlabel('标签中0的比例')
    plt.ylabel('样本数')
    plt.title('每个样本标签补零比例分布')
    plt.tight_layout()
    plt.savefig('label_zero_ratio_hist.png')
    plt.show()
    print('每个样本标签补零比例分布图已保存为 label_zero_ratio_hist.png')

    # 打印补零比例的统计信息
    print(f"标签补零比例均值: {np.mean(zero_ratios):.3f}，中位数: {np.median(zero_ratios):.3f}，最大: {np.max(zero_ratios):.3f}")
    print("如补零比例过高，建议只对真实等位基因位置建模，或采用变长标签模型。")
    # 拆分分箱特征和统计特征
    num_bins = 100
    stat_dim = 9
    X_bins = X[:, :num_bins].reshape(-1, num_bins, 1)
    X_stat = X[:, num_bins:]
    X_train_bins = X_train[:, :num_bins].reshape(-1, num_bins, 1)
    X_train_stat = X_train[:, num_bins:]
    X_test_bins = X_test[:, :num_bins].reshape(-1, num_bins, 1)
    X_test_stat = X_test[:, num_bins:]
    # CNN模型
    cnn_model = create_cnn_model(input_dim=X_train.shape[1], output_dim=y.shape[1], num_bins=num_bins, stat_dim=stat_dim)
    early_stop_cnn = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    history_cnn = cnn_model.fit(
        [X_train_bins, X_train_stat], y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop_cnn],
        verbose=1
    )
    y_pred_cnn = cnn_model.predict([X_test_bins, X_test_stat])
    y_pred_inv_cnn = y_scaler.inverse_transform(y_pred_cnn)
    allele_acc_cnn, genotype_acc_cnn = evaluate_predictions(y_test_inv, y_pred_inv_cnn, threshold=1.0)
    print(f"CNN模型准确率：等位基因级别={allele_acc_cnn:.4f}，基因型级别={genotype_acc_cnn:.4f}")
    cnn_model.save('str_genotype_predictor_cnn.h5')
    # CNN结果保存
    result_rows_cnn = []
    for i in range(len(y_test_inv)):
        result_rows_cnn.append({
            '样本名': groups_test[i],
            '真实基因型': ','.join(map(str, np.round(y_test_inv[i], 2))),
            '预测基因型': ','.join(map(str, np.round(y_pred_inv_cnn[i], 2)))
        })
    pd.DataFrame(result_rows_cnn).to_csv('prediction_results_cnn.csv', index=False, encoding='utf-8-sig')
    with open('evaluation_results_cnn.txt', 'w', encoding='utf-8') as f:
        f.write(f"CNN模型等位基因级别准确率: {allele_acc_cnn:.4f}\n")
        f.write(f"CNN模型基因型级别准确率: {genotype_acc_cnn:.4f}\n")
    # 可视化CNN训练过程
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_cnn.history['loss'], label='训练损失')
    plt.plot(history_cnn.history['val_loss'], label='验证损失')
    plt.title('CNN模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history_cnn.history['mae'], label='训练MAE')
    plt.plot(history_cnn.history['val_mae'], label='验证MAE')
    plt.title('CNN模型平均绝对误差')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    plt.show()
    print('CNN模型训练过程可视化已保存为 cnn_training_history.png')

if __name__ == "__main__":
    main()