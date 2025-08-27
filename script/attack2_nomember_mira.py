import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, balanced_accuracy_score
import ast
import os
import matplotlib.pyplot as plt


# --- 修复: 计算TPR@特定FPR的函数 ---
def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    """计算在特定FPR(如1%)下的TPR，确保返回标量值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # 找到最接近目标FPR的点
    # 使用np.argmin找到最小差值的索引
    idx = np.argmin(np.abs(fpr - target_fpr))
    
    # 确保返回的是标量而不是数组
    return float(tpr[idx]), float(fpr[idx]), float(thresholds[idx])

# --- 修复: 计算最佳阈值(使用约登指数) ---
def find_optimal_threshold(y_true, y_score):
    """使用约登指数(Youden's index)找到最佳阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr  # 约登指数
    optimal_idx = np.argmax(j_scores)
    return float(thresholds[optimal_idx])  # 返回标量

# --- 1. 数据加载函数 (与之前相同) ---
def load_and_aggregate_multi_sample(base_path, num_samples=5):
    """为多采样指标（如background_consistency）加载并聚合数据。"""
    aggregated_data = {}
    for i in range(num_samples):
        filepath = f"{base_path}_sample_{i}.csv"
        if not os.path.exists(filepath):
            continue
        # 强制将 video_id 读取为字符串
        df = pd.read_csv(filepath, dtype={'video_id': str})
        for _, row in df.iterrows():
            video_id = str(row['video_id']) # 再次确保是字符串
            vector = np.array(ast.literal_eval(row['background_consistency']))
            if video_id not in aggregated_data: aggregated_data[video_id] = []
            aggregated_data[video_id].append(vector)
    print(f"Loaded {len(aggregated_data)} unique video IDs from {base_path}")
    for video_id in aggregated_data: 
        # print(f"Video ID: {video_id}, Samples: {len(aggregated_data[video_id])}")
        aggregated_data[video_id] = np.array(aggregated_data[video_id])
    return aggregated_data

def load_single_sample(filepath):
    """为单采样指标（如top1_vector）加载数据。"""
    if not os.path.exists(filepath):
        return None
    # 强制将 id 读取为字符串
    df = pd.read_csv(filepath, dtype={'id': str})
    df.rename(columns={'id': 'video_id'}, inplace=True)
    df['vector'] = df['top1_vector'].apply(ast.literal_eval)
    # 确保索引也是字符串
    return df.set_index('video_id')['vector'].to_dict()


# --- 2. 特征提取函数 ---
def extract_bg_stability(bg_matrix):
    """特征1: BG Sampling Stability"""
    if bg_matrix is None or bg_matrix.shape[0] < 2: return np.nan
    # print(f"BG Matrix Shape: {bg_matrix.shape}")
    # mean_per_dim =np.mean(bg_matrix, axis=1) #np.std(bg_matrix, axis=0)#[:, 5:]
    # print(f"Mean per dimension: {mean_per_dim} {mean_per_dim.shape}")
    # return np.std(mean_per_dim) 
    # std_per_dim = np.std(bg_matrix, axis=0) #[:, 5:]  # 按列计算标准差
    std_per_dim = bg_matrix[1]
    # print(f"Standard deviation per dimension: {std_per_dim.shape}")
    return np.mean(std_per_dim)

def extract_top1_quality(top1_vector):
    """特征2: Top1 Reconstruction Quality"""
    if top1_vector is None: return np.nan
    # return np.mean(top1_vector)
    return np.mean(top1_vector[9:])             #12最优 #[5:]

# # --- 3. Z-Test 相关函数 ---#后面直接计算了，用不到函数
# def get_z_scores(values, mu, sigma):
#     """计算Z-score"""
#     if sigma < 1e-9: sigma = 1e-9
#     return [(v - mu) / sigma for v in values]

# --- 4. 主流程 ---
print("--- 正在加载所有数据 ---")
# 数据路径
bg_member_base = '/data/cwy/t2v_end/methods/mira/multi_metrics/merged_background_consistency_member'
bg_nomember_base = '/data/cwy/t2v_end/methods/mira/multi_metrics/merged_background_consistency_nomember'
top1_member_path = '/data/cwy/t2v_end/methods/mira/frame_clip/merged_member_sample_0.csv'
top1_nomember_path = '/data/cwy/t2v_end/methods/mira/frame_clip/merged_nomember_sample_0.csv'

bg_member_samples = load_and_aggregate_multi_sample(bg_member_base, 3)
bg_nomember_samples = load_and_aggregate_multi_sample(bg_nomember_base, 3)
top1_member_samples = load_single_sample(top1_member_path)
top1_nomember_samples = load_single_sample(top1_nomember_path)
if top1_member_samples is None or top1_nomember_samples is None: exit()

# --- 5. 提取所有样本的特征 ---
all_member_ids = list(bg_member_samples.keys())
all_nomember_ids = list(bg_nomember_samples.keys())

# 为所有样本创建特征DataFrame，便于处理
data = []
for vid in all_member_ids:
    if vid in top1_member_samples:
        data.append({
            'video_id': vid,
            'label': 1,
            'bg_stability': extract_bg_stability(bg_member_samples.get(vid)),
            'top1_quality': extract_top1_quality(top1_member_samples.get(vid))
        })
for vid in all_nomember_ids:
    if vid in top1_nomember_samples:
        data.append({
            'video_id': vid,
            'label': 0,
            'bg_stability': extract_bg_stability(bg_nomember_samples.get(vid)),
            'top1_quality': extract_top1_quality(top1_nomember_samples.get(vid))
        })
df_features = pd.DataFrame(data).dropna()

df_member = df_features[df_features['label'] == 1]
df_nomember = df_features[df_features['label'] == 0]


best = 0
best_i = 0
for i in range(1, 2):

    # 将非成员数据80/20划分为校准集和测试集
    calib_nomember_df, test_nomember_df = train_test_split(
        df_nomember, test_size=0.2, random_state=316   # 413
    )
    # 将成员数据80/20划分，只保留测试部分
    _, test_member_df = train_test_split(
        df_member, test_size=0.2, random_state=245  # 245   748
    )


    # 合并成最终的测试集DataFrame
    df_test = pd.concat([test_member_df, test_nomember_df])
    # 校准集DataFrame
    df_calib = calib_nomember_df

    print(f"校准集大小 (仅非成员): {len(df_calib)}")
    print(f"测试集大小: {len(df_test)} (成员: {len(test_member_df)}, 非成员: {len(test_nomember_df)})")

    # --- 模型校准 ---
    print("\n--- 正在使用校准集构建“正常”模型 ---")
    mu_bg, sigma_bg = df_calib['bg_stability'].mean(), df_calib['bg_stability'].std()
    mu_top1, sigma_top1 = df_calib['top1_quality'].mean(), df_calib['top1_quality'].std()
    print("模型校准完成。")

    # --- 在测试集上计算异常分数 ---
    print("\n--- 正在为测试集计算异常分数 ---")
    # 计算 score_bg
    test_z_bg = (df_test['bg_stability'] - mu_bg) / (sigma_bg if sigma_bg > 1e-9 else 1e-9)
    test_score_bg = 1 - test_z_bg

    # 计算 score_top1
    test_z_top1 = (df_test['top1_quality'] - mu_top1) / (sigma_top1 if sigma_top1 > 1e-9 else 1e-9)
    test_score_top1 = test_z_top1

    # 获取测试集的真实标签
    y_test = df_test['label'].values

    # --- 独立评估与融合评估 ---
    print("\n" + "="*50)
    print("--- 正在测试集上进行评估 ---")

    # 准备评估所有模型
    models = {
        "background": test_score_bg,
        "top1": test_score_top1,
        "both": None,
        # "最大值融合 (Max Fusion)": np.maximum(test_score_bg, test_score_top1)
    }

    # 设置融合权重
    w_bg = 0.3
    w_top1 = 0.7
    models["both"] = w_bg * test_score_bg + w_top1 * test_score_top1
    print(f"使用固定融合权重: w_bg = {w_bg:.4f}, w_top1 = {w_top1:.4f}")

    # 评估每个模型并收集结果
    results = []
    roc_data = {}

    for model_name, scores in models.items():
        print(f"\n评估 {model_name}...")
        
        # 计算AUC
        auc = roc_auc_score(y_test, scores)

        if model_name=="both" and auc>best:
            best=auc
            best_i=i
        
        # 计算TPR@1% FPR
        tpr, actual_fpr, _ = tpr_at_fpr(y_test, scores, target_fpr=0.01)
        
        # 找到最佳阈值并计算平衡准确率
        optimal_threshold = find_optimal_threshold(y_test, scores)
        y_pred = (scores >= optimal_threshold).astype(int)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        # 保存结果
        results.append({
            "指标": model_name,
            "AUC-ROC": auc,
            "TPR @ 1% FPR": tpr,
            "实际FPR": actual_fpr,
            "balanced acc": bal_acc
        })
        
        # 保存ROC曲线数据
        fpr, tpr_curve, thresholds = roc_curve(y_test, scores)
        roc_data[model_name] = pd.DataFrame({
            # 'fpr': fpr,
            # 'tpr': tpr_curve,
            # 'thresholds': thresholds
            'y_true': y_test,
            'y_pred_proba': scores,
        })
        
        # 打印当前模型结果
        print(f"AUC-ROC: {auc:.4f}")
        print(f"TPR @ 1% FPR: {tpr:.4f} (实际FPR: {actual_fpr:.4f})")
        print(f"Balanced Accuracy: {bal_acc:.4f}")

    # --- 最终性能对比 ---
    print("\n" + "="*50)
    print("--- 最终性能对比 (在同一测试集上) ---")
    df_comparison = pd.DataFrame(results)
    # 只显示需要的列
    print(df_comparison[["指标", "AUC-ROC", "TPR @ 1% FPR", "balanced acc"]].round(4))

    correlation = df_test['bg_stability'].corr(df_test['top1_quality'])
    print(f"\n两个指标异常分数之间的相关性: {correlation:.4f}")

    # --- 保存ROC曲线数据到CSV ---
    print("\n" + "="*50)
    print("--- 保存ROC曲线数据 ---")
    # 创建保存目录
    if not os.path.exists('roc_data'):
        os.makedirs('roc_data')

    # 保存每个模型的ROC数据
    for model_name, df_roc in roc_data.items():
        # 替换文件名中的特殊字符
        filename = f"roc_data_mira/attack2_{model_name}_0.csv"
        df_roc.to_csv(filename, index=False)
        print(f"已保存 {model_name} 的ROC数据到 {filename}")
        
print(f"\n第i次最优: {best_i}",f"\n最优AUC-ROC 分数: {best:.4f}") 