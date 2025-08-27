# import os
# import ast
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

# # 设备设置
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- 1. 函数定义 (与场景二代码完全相同) ---
# def load_and_aggregate_multi_sample(base_path, num_samples=5):
#     """为多采样指标（如background_consistency）加载并聚合数据。"""
#     aggregated_data = {}
#     for i in range(num_samples):
#         filepath = f"{base_path}_sample_{i}.csv"
#         if not os.path.exists(filepath):
#             continue
#         df = pd.read_csv(filepath, dtype={'video_id': str})
#         for _, row in df.iterrows():
#             video_id = str(row['video_id'])
#             vector = np.array(ast.literal_eval(row['background_consistency']))
#             if video_id not in aggregated_data: aggregated_data[video_id] = []
#             aggregated_data[video_id].append(vector)
#     for video_id in aggregated_data: aggregated_data[video_id] = np.array(aggregated_data[video_id])
#     return aggregated_data

# def load_single_sample(filepath):
#     """为单采样指标（如top1_vector）加载数据。"""
#     if not os.path.exists(filepath):
#         return None
#     df = pd.read_csv(filepath, dtype={'id': str})
#     df.rename(columns={'id': 'video_id'}, inplace=True)
#     df['vector'] = df['top1_vector'].apply(ast.literal_eval)
#     return df.set_index('video_id')['vector'].to_dict()

# def extract_bg_stability_vector(bg_matrix):
#     """提取BG稳定性向量（[2:]部分）"""
#     if bg_matrix is None or bg_matrix.shape[0] < 2:
#         return np.nan
#     std_per_dim = np.std(bg_matrix, axis=0)
#     return std_per_dim[2:]

# def extract_top1_quality_vector(top1_vector):
#     """提取Top1质量向量（[2:]部分）"""
#     if top1_vector is None:
#         return np.nan
#     return top1_vector[2:]

# # --- 2. 主流程：数据加载与特征提取 ---
# print("--- 场景一：监督学习 ---")
# print("\n--- 正在加载数据并提取原始特征 ---")
# # 数据路径
# bg_member_base = '/data/cwy/t2v_end/methods/animatediff/multi_metrics/background_member_25_8'
# bg_nomember_base = '/data/cwy/t2v_end/methods/animatediff/multi_metrics/background_nomember_25_8'
# top1_member_path = '/data/cwy/t2v_end/methods/animatediff/frame_clip/member_25_8_sample_0.csv'
# top1_nomember_path = '/data/cwy/t2v_end/methods/animatediff/frame_clip/nomember_25_8_sample_0.csv'

# bg_member_samples = load_and_aggregate_multi_sample(bg_member_base, 5)
# bg_nomember_samples = load_and_aggregate_multi_sample(bg_nomember_base, 5)
# top1_member_samples = load_single_sample(top1_member_path)
# top1_nomember_samples = load_single_sample(top1_nomember_path)
# if top1_member_samples is None or top1_nomember_samples is None: exit()

# # 提取所有样本的向量特征并拼接
# data = []
# for vid in bg_member_samples.keys():
#     if vid in top1_member_samples:
#         bg_vec = extract_bg_stability_vector(bg_member_samples.get(vid))
#         top1_vec = extract_top1_quality_vector(top1_member_samples.get(vid))
#         if not (np.isnan(bg_vec).any() or np.isnan(top1_vec).any()):
#             # 拼接两个向量作为样本表示
#             combined_vec = np.concatenate([bg_vec, top1_vec])
#             data.append({
#                 'video_id': vid, 
#                 'label': 1,
#                 'combined_vector': combined_vec
#             })

# for vid in bg_nomember_samples.keys():
#     if vid in top1_nomember_samples:
#         bg_vec = extract_bg_stability_vector(bg_nomember_samples.get(vid))
#         top1_vec = extract_top1_quality_vector(top1_nomember_samples.get(vid))
#         if not (np.isnan(bg_vec).any() or np.isnan(top1_vec).any()):
#             combined_vec = np.concatenate([bg_vec, top1_vec])
#             data.append({
#                 'video_id': vid, 
#                 'label': 0,
#                 'combined_vector': combined_vec
#             })

# # 转换为DataFrame并检查向量维度
# df_combined = pd.DataFrame(data)
# if df_combined.empty:
#     raise ValueError("无有效样本数据，请检查特征提取逻辑")

# # 获取输入维度（拼接后的向量长度）
# INPUT_DIM = len(df_combined['combined_vector'].iloc[0])

# # --- 3. MLP模型参数与定义 ---
# MLP_PARAMS = {
#     "input_dim": INPUT_DIM,
#     "hidden_dim1": 64,
#     "hidden_dim2": 8,
#     "epochs": 500,
#     "batch_size": 32,
#     "lr": 0.01,
#     "test_size": 0.2,
#     "random_state": 510
# }

# class FeatureDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels
        
#     def __len__(self):
#         return len(self.features)
        
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.3):
#         super(MLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim1, hidden_dim2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim2, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         return self.layers(x)

# # --- 4. 模型训练与评估 ---
# def train_evaluate_mlp(features, labels):
#     # 数据划分（8:2）
#     X_train, X_test, y_train, y_test = train_test_split(
#         features, labels,
#         test_size=MLP_PARAMS["test_size"],
#         random_state=MLP_PARAMS["random_state"],
#         stratify=labels  # 保持类别分布一致
#     )
    
#     # 特征标准化
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # 转换为张量数据集
#     train_dataset = FeatureDataset(
#         torch.tensor(X_train_scaled, dtype=torch.float32),
#         torch.tensor(y_train, dtype=torch.float32)
#     )
#     test_dataset = FeatureDataset(
#         torch.tensor(X_test_scaled, dtype=torch.float32),
#         torch.tensor(y_test, dtype=torch.float32)
#     )
    
#     # 数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=MLP_PARAMS["batch_size"], shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=MLP_PARAMS["batch_size"], shuffle=False)
    
#     # 初始化模型
#     model = MLP(
#         input_dim=MLP_PARAMS["input_dim"],
#         hidden_dim1=MLP_PARAMS["hidden_dim1"],
#         hidden_dim2=MLP_PARAMS["hidden_dim2"]
#     ).to(DEVICE)
    
#     # 损失函数与优化器（带L2正则化）
#     criterion = nn.BCELoss()
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=MLP_PARAMS["lr"],
#         weight_decay=1e-3
#     )
    
#     # 训练过程（带早停机制）
#     best_val_auc = 0.0
#     patience = 5
#     counter = 0
    
#     for epoch in range(MLP_PARAMS["epochs"]):
#         model.train()
#         total_loss = 0.0
        
#         for batch_features, batch_labels in train_loader:
#             batch_features = batch_features.to(DEVICE)
#             batch_labels = batch_labels.to(DEVICE).unsqueeze(1)
            
#             optimizer.zero_grad()
#             outputs = model(batch_features)
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         # 验证阶段
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE))
#             val_proba = val_outputs.cpu().numpy().flatten()
#             val_auc = roc_auc_score(y_test, val_proba)
        
#         # 早停判断
#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             counter = 0
#             torch.save(model.state_dict(), "best_model.pth")
#         else:
#             counter += 1
#             if counter >= patience and best_val_auc > 0.93:
#                 print(f"早停于第{epoch+1}轮，最佳验证AUC: {best_val_auc:.4f}")
#                 break
        
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/{MLP_PARAMS['epochs']}], Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
    
#     # 加载最佳模型评估
#     model.load_state_dict(torch.load("best_model.pth"))
#     model.eval()
    
#     y_pred_proba = []
#     y_true = []
#     with torch.no_grad():
#         for batch_features, batch_labels in test_loader:
#             batch_features = batch_features.to(DEVICE)
#             outputs = model(batch_features)
#             y_pred_proba.extend(outputs.cpu().numpy().flatten())
#             y_true.extend(batch_labels.numpy().flatten())
    
#     # 计算评估指标
#     y_pred_proba = np.array(y_pred_proba)
#     y_true = np.array(y_true)
    
#     # 寻找最优阈值（约登指数）
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
#     j_scores = tpr - fpr
#     optimal_threshold = thresholds[np.argmax(j_scores)]
#     y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
#     # 输出结果
#     print("\n=== MLP模型评估结果 ===")
#     print(f"最优阈值: {optimal_threshold:.4f}")
#     print(f"测试集准确率: {accuracy_score(y_true, y_pred):.4f}")
#     print(f"测试集AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")

# # --- 5. 执行训练与评估 ---
# # 提取特征和标签
# X = np.stack(df_combined['combined_vector'].values)
# y = df_combined['label'].values

# # 启动训练评估流程
# print("\n--- 开始MLP模型训练与评估 ---")
# train_evaluate_mlp(X, y)


#方法一：利用MLP可以达到AUC93.7%，ACC87.5%
#可以调整参数进行优化，hidden_dim1和2；std_per_dim[2:]这里向量维度保留越多越好



import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

# 设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SELECT_SIZE = 4  # 从第几个维度开始选择特征
best_results = 0.0
best_output = {}

# 通用参数设置
base_params = {
    "epochs": 500,
    "batch_size": 32,
    "lr": 0.01,
    "test_size": 0.2,
    "random_state": 510,  # 510  730
    "hidden_dim1": 64,
    "hidden_dim2": 8
}


# --- 1. 函数定义 ---
def load_and_aggregate_multi_sample(base_path, num_samples=5):
    """为多采样指标（如background_consistency）加载并聚合数据。"""
    aggregated_data = {}
    for i in range(num_samples):
        filepath = f"{base_path}_sample_{i}.csv"
        if not os.path.exists(filepath):
            continue
        df = pd.read_csv(filepath, dtype={'video_id': str})
        for _, row in df.iterrows():
            video_id = str(row['video_id'])
            vector = np.array(ast.literal_eval(row['background_consistency']))
            if video_id not in aggregated_data: aggregated_data[video_id] = []
            aggregated_data[video_id].append(vector)
    for video_id in aggregated_data: aggregated_data[video_id] = np.array(aggregated_data[video_id])
    return aggregated_data

def load_single_sample(filepath):
    """为单采样指标（如top1_vector）加载数据。"""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, dtype={'id': str})
    df.rename(columns={'id': 'video_id'}, inplace=True)
    df['vector'] = df['top1_vector'].apply(ast.literal_eval)  # top1_vector  cross_similarities_vector
    return df.set_index('video_id')['vector'].to_dict()

def extract_bg_stability_vector(bg_matrix):
    """提取BG稳定性向量（[2:]部分）"""
    if bg_matrix is None or bg_matrix.shape[0] < 2:
        return np.nan
    std_per_dim = np.std(bg_matrix, axis=0)
    return std_per_dim[SELECT_SIZE:]
    # return std_per_dim[0:5]
    # return std_per_dim[5:10]
    # return std_per_dim[10:15]
    

def extract_top1_quality_vector(top1_vector):
    """提取Top1质量向量（[2:]部分）"""
    if top1_vector is None:
        return np.nan
    return top1_vector[SELECT_SIZE:]
    # return top1_vector[1:6]
    # return top1_vector[6:11]
    # return top1_vector[11:16]

def calculate_tpr_at_fpr(y_true, y_pred_proba, target_fpr=0.05):
    """计算在特定FPR（如0.1%）下的TPR值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # 找到最接近目标FPR的阈值
    idx = np.argmin(np.abs(fpr - target_fpr))
    
    # 如果找到的FPR小于目标，尝试找到下一个更大的FPR
    if fpr[idx] < target_fpr and idx < len(fpr) - 1:
        idx += 1
    
    return tpr[idx], fpr[idx]

def calculate_balanced_accuracy(y_true, y_pred):
    """计算平衡准确率：(灵敏度 + 特异度) / 2"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return (sensitivity + specificity) / 2

# --- 2. 数据加载与特征提取 ---
print("--- 场景一：监督学习 ---")
print("\n--- 正在加载数据并提取原始特征 ---")

# dynamic_degree   subject_consistency  background background_consistency
temp_name = "background"
# 数据路径
bg_member_base = f'/data/cwy/t2v_end/methods/animatediff/multi_metrics/{temp_name}_member_25_8'
bg_nomember_base = f'/data/cwy/t2v_end/methods/animatediff/multi_metrics/{temp_name}_nomember_25_8'
top1_member_path = '/data/cwy/t2v_end/methods/animatediff/frame_clip/member_25_8_sample_0.csv'
top1_nomember_path = '/data/cwy/t2v_end/methods/animatediff/frame_clip/nomember_25_8_sample_0.csv'

# 加载所有数据
bg_member_samples = load_and_aggregate_multi_sample(bg_member_base, 5)
bg_nomember_samples = load_and_aggregate_multi_sample(bg_nomember_base, 5)
top1_member_samples = load_single_sample(top1_member_path)
top1_nomember_samples = load_single_sample(top1_nomember_path)

if top1_member_samples is None or top1_nomember_samples is None:
    print("无法加载top1特征数据，程序退出")
    exit()

# 提取所有样本的向量特征（三种情况）
data_bg = []  # 仅使用background特征
data_top1 = []  # 仅使用top1特征
data_combined = []  # 两种特征结合

# 处理member样本
for vid in bg_member_samples.keys():
    if vid in top1_member_samples:
        bg_vec = extract_bg_stability_vector(bg_member_samples.get(vid))
        top1_vec = extract_top1_quality_vector(top1_member_samples.get(vid))
        
        # 检查向量有效性
        valid_bg = not np.isnan(bg_vec).any()
        valid_top1 = not np.isnan(top1_vec).any()
        
        if valid_bg:
            data_bg.append({
                'video_id': vid, 
                'label': 1,
                'vector': bg_vec
            })
        
        if valid_top1:
            data_top1.append({
                'video_id': vid, 
                'label': 1,
                'vector': top1_vec
            })
        
        if valid_bg and valid_top1:
            combined_vec = np.concatenate([bg_vec, top1_vec])
            data_combined.append({
                'video_id': vid, 
                'label': 1,
                'vector': combined_vec
            })

# 处理non-member样本
for vid in bg_nomember_samples.keys():
    if vid in top1_nomember_samples:
        bg_vec = extract_bg_stability_vector(bg_nomember_samples.get(vid))
        top1_vec = extract_top1_quality_vector(top1_nomember_samples.get(vid))
        
        # 检查向量有效性
        valid_bg = not np.isnan(bg_vec).any()
        valid_top1 = not np.isnan(top1_vec).any()
        
        if valid_bg:
            data_bg.append({
                'video_id': vid, 
                'label': 0,
                'vector': bg_vec
            })
        
        if valid_top1:
            data_top1.append({
                'video_id': vid, 
                'label': 0,
                'vector': top1_vec
            })
        
        if valid_bg and valid_top1:
            combined_vec = np.concatenate([bg_vec, top1_vec])
            data_combined.append({
                'video_id': vid, 
                'label': 0,
                'vector': combined_vec
            })

# 转换为DataFrame并检查数据有效性
datasets = {
    # "background": pd.DataFrame(data_bg),
    # "top1": pd.DataFrame(data_top1),
    "both": pd.DataFrame(data_combined)
}

# 检查每个数据集是否为空
for name, df in datasets.items():
    if df.empty:
        print(f"警告: {name}的数据集为空，请检查特征提取逻辑")

# --- 3. MLP模型参数与定义 ---
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

# --- 4. 模型训练与评估 ---
def train_evaluate_mlp(features, labels, input_dim, model_name, params):
    """
    训练和评估MLP模型，输出AUC、TPR@0.1%FPR和平衡准确率
    
    参数:
        features: 输入特征
        labels: 标签
        input_dim: 输入维度
        model_name: 模型名称，用于输出
        params: 模型参数字典
    """
    print(f"\n--- 开始{model_name}的MLP模型训练与评估 ---")
    
    # 数据划分（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=labels  # 保持类别分布一致
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 转换为张量数据集
    train_dataset = FeatureDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = FeatureDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
    
    # 初始化模型
    model = MLP(
        input_dim=input_dim,
        hidden_dim1=params["hidden_dim1"],
        hidden_dim2=params["hidden_dim2"]
    ).to(DEVICE)
    
    # 损失函数与优化器（带L2正则化）
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=1e-3
    )
    
    # 训练过程（带早停机制）
    best_val_auc = 0.0
    patience = 5
    counter = 0
    model_save_path = f"best_model_{model_name.replace(' ', '_')}.pth"
    
    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE))
            val_proba = val_outputs.cpu().numpy().flatten()
            val_auc = roc_auc_score(y_test, val_proba)
        
        # 早停判断
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            counter += 1
            if counter >= patience and best_val_auc > 0.93:
                print(f"早停于第{epoch+1}轮，最佳验证AUC: {best_val_auc:.4f}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{params['epochs']}], Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
    
    # 加载最佳模型评估
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    y_pred_proba = []
    y_true = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(DEVICE)
            outputs = model(batch_features)
            y_pred_proba.extend(outputs.cpu().numpy().flatten())
            y_true.extend(batch_labels.numpy().flatten())
    
    # 计算评估指标
    y_pred_proba = np.array(y_pred_proba)
    y_true = np.array(y_true)
    
    # 1. 计算AUC
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    # 2. 计算TPR@0.1% FPR
    tpr_at_target, actual_fpr = calculate_tpr_at_fpr(y_true, y_pred_proba, target_fpr=0.01)
    
    # 3. 寻找最优阈值（约登指数）并计算平衡准确率
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_threshold = thresholds[np.argmax(j_scores)]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    balanced_acc = calculate_balanced_accuracy(y_true, y_pred)
    
    # 输出结果
    print(f"\n=== {model_name}的MLP模型评估结果 ===")
    print(f"样本数量: {len(features)}")
    print(f"特征维度: {input_dim}")
    print(f"最优阈值: {optimal_threshold:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"TPR @ {actual_fpr*100:.3f}% FPR: {tpr_at_target:.4f}")
    print(f"平衡准确率 (Balanced Accuracy): {balanced_acc:.4f}")
    
    return {
        "model_name": model_name,
        "sample_count": len(features),
        "feature_dim": input_dim,
        "threshold": optimal_threshold,
        "auc": auc_score,
        "tpr_at_0.1pct_fpr": tpr_at_target,
        "actual_fpr": actual_fpr,
        "balanced_accuracy": balanced_acc,
        "y_true": y_true,          # 测试集真实标签
        "y_pred_proba": y_pred_proba  # 测试集预测概率
    }

# --- 5. 执行训练与评估 ---
for index in range(5):
    # 存储所有结果
    all_results = []
    # 存储ROC专用数据（真实标签+预测概率）
    roc_data = {}

    # 分别训练和评估三种情况
    for name, df in datasets.items():
        if df.empty:
            print(f"跳过{name}的训练，因为数据集为空")
            continue
            
        # 获取特征和标签
        X = np.stack(df['vector'].values)
        y = df['label'].values
        input_dim = len(df['vector'].iloc[0])
        
        # 训练评估模型
        result = train_evaluate_mlp(X, y, input_dim, name, base_params)
        all_results.append(result)

        # 保存ROC数据到专用字典
        roc_data[name] = {
            "y_true": result["y_true"],
            "y_pred_proba": result["y_pred_proba"]
        }


    # 汇总输出所有结果
    print("\n" + "="*70)
    print("所有特征组合的模型评估结果汇总")
    print("="*70)
    print(f"{'模型名称':<20} {'AUC':<10} {'TPR@1%FPR':<15} {'平衡准确率':<15}")
    print("-"*70)
    for res in all_results:
        print(f"{res['model_name']:<20} {res['auc']:.4f}     {res['tpr_at_0.1pct_fpr']:.4f}        {res['balanced_accuracy']:.4f}")

    if res['auc'] > best_results:
        best_results = res['auc']
        best_output = res
        # --- 新增：输出ROC曲线所需数据 ---
        print("\n" + "="*70)
        print("ROC曲线绘制数据（按特征组合输出）")
        print("="*70)
        for model_name, data in roc_data.items():
            print(f"\n--- {model_name}的ROC数据 ---")
            print(f"真实标签 (y_true) 前10个样本: {data['y_true'][:10]}")  # 打印前10个示例
            print(f"预测概率 (y_pred_proba) 前10个样本: {data['y_pred_proba'][:10].round(4)}")
            print(f"数据总长度: {len(data['y_true'])} 个样本")
            
            # 可选：保存为CSV文件（方便后续可视化工具读取）
            roc_df = pd.DataFrame({
                "y_true": data["y_true"],
                "y_pred_proba": data["y_pred_proba"]
            })
            csv_path = f"roc_data/attack1_{model_name.replace(' ', '_')}_{SELECT_SIZE}.csv"
            # roc_df.to_csv(csv_path, index=False)
            print(f"ROC数据已保存至: {csv_path}")

print(f"\n最佳模型AUC: {best_results:.4f}")
print(f" {best_output['auc']:.4f}     {best_output['tpr_at_0.1pct_fpr']:.4f}        {best_output['balanced_accuracy']:.4f}")


# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap  # 需额外安装：pip install umap-learn

# # 设置中文字体显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# def visualize_distribution(features, labels, method="pca", title="成员与非成员特征分布"):
#     """
#     可视化成员和非成员的特征分布（降维到2D）
    
#     参数:
#         features: 高维特征向量 (n_samples, n_features)
#         labels: 标签 (0=非成员, 1=成员)
#         method: 降维方法 ("pca", "tsne", "umap")
#         title: 图表标题
#     """
#     # 特征标准化（与模型训练保持一致）
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
    
#     # 降维处理
#     if method == "pca":
#         # PCA：保留全局结构，计算快
#         reducer = PCA(n_components=2, random_state=base_params["random_state"])
#         embeddings = reducer.fit_transform(features_scaled)
#         print(f"PCA解释方差比: {reducer.explained_variance_ratio_} (累计: {sum(reducer.explained_variance_ratio_):.2f})")
#     elif method == "tsne":
#         # t-SNE：保留局部结构，适合非线性分布
#         reducer = TSNE(n_components=2, perplexity=30, random_state=base_params["random_state"])
#         embeddings = reducer.fit_transform(features_scaled)
#     elif method == "umap":
#         # UMAP：平衡全局和局部结构
#         reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=base_params["random_state"])
#         embeddings = reducer.fit_transform(features_scaled)
#     else:
#         raise ValueError("降维方法需为 'pca', 'tsne' 或 'umap'")
    
#     # 转换为DataFrame方便绘图
#     df_embedding = pd.DataFrame({
#         "维度1": embeddings[:, 0],
#         "维度2": embeddings[:, 1],
#         "类别": ["成员" if label == 1 else "非成员" for label in labels]
#     })
    
#     # 绘制散点图
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(
#         data=df_embedding,
#         x="维度1",
#         y="维度2",
#         hue="类别",
#         palette={"成员": "#1f77b4", "非成员": "#ff7f0e"},  # 蓝色=成员，橙色=非成员
#         alpha=0.7,  # 点透明度
#         s=50,  # 点大小
#         edgecolor="k",  # 点边缘颜色
#         linewidth=0.5  # 点边缘宽度
#     )
    
#     # 添加标题和标签
#     plt.title(f"{title} ({method.upper()}降维)", fontsize=15)
#     plt.xlabel(f"{method.upper()} 维度1", fontsize=12)
#     plt.ylabel(f"{method.upper()} 维度2", fontsize=12)
#     plt.legend(title="类别", fontsize=10, title_fontsize=12)
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"member_nonmember_distribution_{method}.png", dpi=300)  # 保存图片
#     plt.show()


# # --- 执行可视化 ---
# if "both" in datasets and not datasets["both"].empty:
#     # 提取"both"特征组合的特征和标签
#     df_both = datasets["both"]
#     X_both = np.stack(df_both["vector"].values)  # 特征向量
#     y_both = df_both["label"].values  # 标签（1=成员，0=非成员）
    
#     # 分别用PCA、t-SNE、UMAP可视化
#     visualize_distribution(
#         features=X_both,
#         labels=y_both,
#         method="pca",
#         title="成员与非成员的组合特征分布"
#     )
    
#     visualize_distribution(
#         features=X_both,
#         labels=y_both,
#         method="tsne",
#         title="成员与非成员的组合特征分布"
#     )
    
#     visualize_distribution(
#         features=X_both,
#         labels=y_both,
#         method="umap",
#         title="成员与非成员的组合特征分布"
#     )
# else:
#     print("无法可视化：'both'数据集为空")





# # #方法二：能达到91.94%左右,逻辑回归，标量的形式
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# import lightgbm as lgb
# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
# import ast
# import os

# # --- 1. 函数定义 (与场景二代码完全相同) ---
# def load_and_aggregate_multi_sample(base_path, num_samples=5):
#     """为多采样指标（如background_consistency）加载并聚合数据。"""
#     aggregated_data = {}
#     for i in range(num_samples):
#         filepath = f"{base_path}_sample_{i}.csv"
#         if not os.path.exists(filepath):
#             continue
#         df = pd.read_csv(filepath, dtype={'video_id': str})
#         for _, row in df.iterrows():
#             video_id = str(row['video_id'])
#             vector = np.array(ast.literal_eval(row['background_consistency']))
#             if video_id not in aggregated_data: aggregated_data[video_id] = []
#             aggregated_data[video_id].append(vector)
#     for video_id in aggregated_data: aggregated_data[video_id] = np.array(aggregated_data[video_id])
#     return aggregated_data

# def load_single_sample(filepath):
#     """为单采样指标（如top1_vector）加载数据。"""
#     if not os.path.exists(filepath):
#         return None
#     df = pd.read_csv(filepath, dtype={'id': str})
#     df.rename(columns={'id': 'video_id'}, inplace=True)
#     df['vector'] = df['top1_vector'].apply(ast.literal_eval)
#     return df.set_index('video_id')['vector'].to_dict()

# def extract_bg_stability(bg_matrix):
#     """特征1: BG Sampling Stability"""
#     if bg_matrix is None or bg_matrix.shape[0] < 2: return np.nan
#     mean_per_dim = np.mean(bg_matrix, axis=1) #np.std(bg_matrix, axis=0)#
#     return np.std(mean_per_dim)

# def extract_top1_quality(top1_vector):
#     """特征2: Top1 Reconstruction Quality"""
#     if top1_vector is None: return np.nan
#     return np.mean(top1_vector[8:])

# # --- 2. 主流程：数据加载与特征提取 ---
# print("--- 场景一：监督学习 ---")
# print("\n--- 正在加载数据并提取原始特征 ---")
# # 数据路径
# bg_member_base = '/data/cwy/t2v_end/methods/animatediff/multi_metrics/background_member_25_8'
# bg_nomember_base = '/data/cwy/t2v_end/methods/animatediff/multi_metrics/background_nomember_25_8'
# top1_member_path = '/data/cwy/t2v_end/methods/animatediff/frame_clip/member_25_8_sample_0.csv'
# top1_nomember_path = '/data/cwy/t2v_end/methods/animatediff/frame_clip/nomember_25_8_sample_0.csv'

# bg_member_samples = load_and_aggregate_multi_sample(bg_member_base, 5)
# bg_nomember_samples = load_and_aggregate_multi_sample(bg_nomember_base, 5)
# top1_member_samples = load_single_sample(top1_member_path)
# top1_nomember_samples = load_single_sample(top1_nomember_path)
# if top1_member_samples is None or top1_nomember_samples is None: exit()

# # 提取所有样本的特征
# data = []
# for vid in bg_member_samples.keys():
#     if vid in top1_member_samples:
#         data.append({
#             'video_id': vid, 'label': 1,
#             'bg_stability': extract_bg_stability(bg_member_samples.get(vid)),
#             'top1_quality': extract_top1_quality(top1_member_samples.get(vid))
#         })
# for vid in bg_nomember_samples.keys():
#     if vid in top1_nomember_samples:
#         data.append({
#             'video_id': vid, 'label': 0,
#             'bg_stability': extract_bg_stability(bg_nomember_samples.get(vid)),
#             'top1_quality': extract_top1_quality(top1_nomember_samples.get(vid))
#         })
# df_features = pd.DataFrame(data).dropna()

# # --- 3. 监督学习流程 ---
# print("\n" + "="*50)
# print("--- 正在训练和评估监督学习模型 ---")

# # 准备特征矩阵 X 和标签 y
# # 注意：我们直接使用原始特征，因为监督模型可以自己学习它们的方向和尺度
# X = df_features[['bg_stability', 'top1_quality']].values
# y = df_features['label'].values

# # best=0
# # print("\n开始进行1000次随机划分的交叉验证...")
# # for i in range(1, 1001):


# # 将所有数据（成员+非成员）80/20划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=510, stratify=y)
# print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
# print(f"训练集中标签分布:\n{pd.Series(y_train).value_counts()}")

# # 特征缩放
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # --- a) 训练逻辑回归模型 ---
# print("\n--- 训练逻辑回归模型 ---")
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X_train_scaled, y_train)

# # 评估
# lr_predictions = log_reg.predict(X_test_scaled)
# lr_probabilities = log_reg.predict_proba(X_test_scaled)[:, 1]
# lr_auc = roc_auc_score(y_test, lr_probabilities)
# print(f"逻辑回归 - AUC-ROC 分数: {lr_auc:.4f}")
# print("逻辑回归 - 分类报告:")
# print(classification_report(y_test, lr_predictions, target_names=['Non-Member', 'Member']))


#     if lr_auc>best:
#         best=lr_auc
#         best_i=i
# print(f"\n第i次最优: {best_i}",f"\n最优AUC-ROC 分数: {best:.4f}")




# #逻辑回归，基于向量，仅87%
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
# from sklearn.linear_model import LogisticRegression
# import ast
# import os
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # --- 1. 函数定义 (与场景二代码完全相同) ---
# def load_and_aggregate_multi_sample(base_path, num_samples=5):
#     """为多采样指标（如background_consistency）加载并聚合数据。"""
#     aggregated_data = {}
#     for i in range(num_samples):
#         filepath = f"{base_path}_sample_{i}.csv"
#         if not os.path.exists(filepath):
#             continue
#         df = pd.read_csv(filepath, dtype={'video_id': str})
#         for _, row in df.iterrows():
#             video_id = str(row['video_id'])
#             vector = np.array(ast.literal_eval(row['background_consistency']))
#             if video_id not in aggregated_data: aggregated_data[video_id] = []
#             aggregated_data[video_id].append(vector)
#     for video_id in aggregated_data: aggregated_data[video_id] = np.array(aggregated_data[video_id])
#     return aggregated_data

# def load_single_sample(filepath):
#     """为单采样指标（如top1_vector）加载数据。"""
#     if not os.path.exists(filepath):
#         return None
#     df = pd.read_csv(filepath, dtype={'id': str})
#     df.rename(columns={'id': 'video_id'}, inplace=True)
#     df['vector'] = df['top1_vector'].apply(ast.literal_eval)
#     return df.set_index('video_id')['vector'].to_dict()

# # --- 2. 优化后的特征提取函数 ---
# def extract_bg_stability_vector(bg_matrix):
#     """特征向量1: BG Sampling Stability Vector (15-dim)"""
#     if bg_matrix is None or bg_matrix.shape[0] < 2:
#         return np.nan
#     std_per_dim = np.std(bg_matrix, axis=0)
#     return std_per_dim[2:]

# def extract_top1_quality_vector(top1_vector):
#     """特征向量2: Top1 Reconstruction Quality Vector (16-dim)"""
#     if top1_vector is None:
#         return np.nan
#     return top1_vector[2:]

# # --- 3. 主流程：数据加载与特征提取 ---
# print("--- 场景一：监督学习 ---")
# print("\n--- 正在加载数据并提取原始特征 ---")
# # 数据路径
# bg_member_base = '/data/cwy/t2v_end/methods/animatediff/multi_metrics/background_member_25_8'
# bg_nomember_base = '/data/cwy/t2v_end/methods/animatediff/multi_metrics/background_nomember_25_8'
# top1_member_path = '/data/cwy/t2v_end/methods/animatediff/frame_clip/member_25_8_sample_0.csv'
# top1_nomember_path = '/data/cwy/t2v_end/methods/animatediff/frame_clip/nomember_25_8_sample_0.csv'

# bg_member_samples = load_and_aggregate_multi_sample(bg_member_base, 5)
# bg_nomember_samples = load_and_aggregate_multi_sample(bg_nomember_base, 5)
# top1_member_samples = load_single_sample(top1_member_path)
# top1_nomember_samples = load_single_sample(top1_nomember_path)
# if top1_member_samples is None or top1_nomember_samples is None: exit()

# # 提取所有样本的向量特征并拼接
# data = []
# for vid in bg_member_samples.keys():
#     if vid in top1_member_samples:
#         bg_vec = extract_bg_stability_vector(bg_member_samples.get(vid))
#         top1_vec = extract_top1_quality_vector(top1_member_samples.get(vid))
#         if not (np.isnan(bg_vec).any() or np.isnan(top1_vec).any()):
#             # 拼接两个向量作为样本表示
#             combined_vec = np.concatenate([bg_vec, top1_vec])
#             data.append({
#                 'video_id': vid, 
#                 'label': 1,
#                 'combined_vector': combined_vec
#             })

# for vid in bg_nomember_samples.keys():
#     if vid in top1_nomember_samples:
#         bg_vec = extract_bg_stability_vector(bg_nomember_samples.get(vid))
#         top1_vec = extract_top1_quality_vector(top1_nomember_samples.get(vid))
#         if not (np.isnan(bg_vec).any() or np.isnan(top1_vec).any()):
#             combined_vec = np.concatenate([bg_vec, top1_vec])
#             data.append({
#                 'video_id': vid, 
#                 'label': 0,
#                 'combined_vector': combined_vec
#             })

# # 转换为DataFrame并检查向量维度
# df_combined = pd.DataFrame(data)
# if df_combined.empty:
#     raise ValueError("无有效样本数据，请检查特征提取逻辑")


# # --- 3. 监督学习流程 ---
# print("\n" + "="*50)
# print("--- 正在训练和评估监督学习模型 ---")

# # 提取特征和标签
# X = np.stack(df_combined['combined_vector'].values)
# y = df_combined['label'].values
# # 将所有数据划分为训练集和测试集
# # stratify=y 确保划分后训练集和测试集中的类别比例与原始数据一致
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
# print(f"训练集中标签分布:\n{pd.Series(y_train).value_counts()}")

# # 特征缩放
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # --- a) 训练逻辑回归模型 (作为基准) ---
# print("\n--- 训练逻辑回归模型 ---")
# log_reg = LogisticRegression(random_state=42, max_iter=1000) # 增加max_iter以确保收敛
# log_reg.fit(X_train_scaled, y_train)
# lr_probabilities = log_reg.predict_proba(X_test_scaled)[:, 1]
# lr_auc = roc_auc_score(y_test, lr_probabilities)
# print(f"逻辑回归 - AUC-ROC 分数: {lr_auc:.4f}")






