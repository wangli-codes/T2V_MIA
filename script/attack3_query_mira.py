# 方法一与方法二相同，background还是先计算每个样本的均值，再求标准差

import os
import ast
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.metrics import (roc_auc_score, roc_curve, 
                             balanced_accuracy_score)
import matplotlib.pyplot as plt


# --- 1. 数据加载函数 ---
def load_and_aggregate_multi_sample(base_path, num_samples=5) -> Dict[str, np.ndarray]:
    """加载多采样指标（如background_consistency）并聚合为矩阵"""
    aggregated_data = {}
    for i in range(num_samples):
        filepath = f"{base_path}_sample_{i}.csv"
        if not os.path.exists(filepath):
            print(f"警告：文件不存在 {filepath}")
            continue
        df = pd.read_csv(filepath, dtype={'video_id': str})
        for _, row in df.iterrows():
            video_id = str(row['video_id'])
            vector = np.array(ast.literal_eval(row['background_consistency']))
            if video_id not in aggregated_data:
                aggregated_data[video_id] = []
            aggregated_data[video_id].append(vector)
    # 转换为numpy数组（shape: [num_samples, vector_dim]）
    for video_id in aggregated_data:
        aggregated_data[video_id] = np.array(aggregated_data[video_id])
    return aggregated_data


def load_and_aggregate_top1_multi_sample(base_path, num_samples=5) -> Dict[str, np.ndarray]:
    """加载多采样的top1_vector并聚合为矩阵（适配5次采样）"""
    aggregated_data = {}
    for i in range(num_samples):
        filepath = f"{base_path}_sample_{i}.csv"
        if not os.path.exists(filepath):
            print(f"警告：文件不存在 {filepath}")
            continue
        df = pd.read_csv(filepath, dtype={'id': str})
        df.rename(columns={'id': 'video_id'}, inplace=True)
        for _, row in df.iterrows():
            video_id = str(row['video_id'])
            vector = np.array(ast.literal_eval(row['top1_vector']))
            if video_id not in aggregated_data:
                aggregated_data[video_id] = []
            aggregated_data[video_id].append(vector)
    # 转换为numpy数组（shape: [num_samples, vector_dim]）
    for video_id in aggregated_data:
        aggregated_data[video_id] = np.array(aggregated_data[video_id])
    return aggregated_data


# --- 2. 特征提取函数 ---
def extract_bg_stability(bg_matrix):
    """特征1: BG Sampling Stability"""
    if bg_matrix is None or bg_matrix.shape[0] < 2: return np.nan
    # std_per_dim = np.std(bg_matrix, axis=0)
    std_per_dim = bg_matrix[1]
    # print(std_per_dim)
    return 1-np.mean(std_per_dim[:8])#[2:5]
    # std_per_dim =np.mean(bg_matrix[:, 4:], axis=1) #np.std(bg_matrix, axis=0)#
    # return 1-np.std(std_per_dim)#np.mean(std_per_dim)-3*np.std(std_per_dim)

# def extract_top1_quality(top1_vector):
#     """特征2: Top1 Reconstruction Quality"""
#     if top1_vector is None or top1_vector.shape[0] < 2: return np.nan
#     mean_top1_vector =np.mean(top1_vector, axis=1)
#     return np.mean(mean_top1_vector)#12最优

def extract_top1_quality(top1_vector):
    """特征2: Top1 Reconstruction Quality"""
    if top1_vector is None: return np.nan
    return np.mean(top1_vector[:, 9:])             #12最优 #[5:]



# --- 3. 评估指标计算函数 ---
def calculate_tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    """计算特定FPR（如1%）对应的TPR"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # 找到最接近目标FPR的索引
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx]

def calculate_metrics(y_true, y_score):
    """计算所有评估指标"""
    # AUC-ROC
    auc = roc_auc_score(y_true, y_score)
    
    # TPR @ 1% FPR
    tpr_at_1pct_fpr = calculate_tpr_at_fpr(y_true, y_score, 0.01)
    
    # 平衡准确率
    # 先找到最佳阈值（约登指数）
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # 使用最佳阈值计算预测标签
    y_pred = (y_score >= best_threshold).astype(int)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    return {
        "AUC-ROC": auc,
        "TPR @ 1% FPR": tpr_at_1pct_fpr,
        "balanced_acc": balanced_acc,
        "best_threshold": best_threshold
    }


# --- 4. 结果保存函数 ---
def save_roc_data(y_true, y_pred_proba, score_name, output_dir):
    """保存特定分数的ROC数据到CSV文件"""
    roc_data = pd.DataFrame({
        "y_true": y_true,
        "y_pred_proba": y_pred_proba
    })
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存数据
    output_path = os.path.join(output_dir, f"attack3_{score_name}_0.csv")
    roc_data.to_csv(output_path, index=False)
    print(f"{score_name}的ROC曲线数据已保存至: {output_path}")
    return output_path

# --- 3. 主流程 ---

# 数据路径（5次采样的基础路径）
bg_member_base = '/data/cwy/t2v_end/methods/mira/multi_metrics/merged_background_consistency_member'
bg_nomember_base = '/data/cwy/t2v_end/methods/mira/multi_metrics/merged_background_consistency_nomember'
top1_member_base = '/data/cwy/t2v_end/methods/mira/frame_clip/merged_member'  # 基础路径（不含_sample_i）
top1_nomember_base = '/data/cwy/t2v_end/methods/mira/frame_clip/merged_nomember'  # 基础路径

# 加载数据（5次采样）
print("=== 加载member数据 ===")
bg_member = load_and_aggregate_multi_sample(bg_member_base, 3)
top1_member = load_and_aggregate_top1_multi_sample(top1_member_base, 1)

print("\n=== 加载nomember数据 ===")
bg_nomember = load_and_aggregate_multi_sample(bg_nomember_base, 3)
top1_nomember = load_and_aggregate_top1_multi_sample(top1_nomember_base, 1)
if top1_member is None or top1_nomember is None: exit()


# 为所有样本创建特征DataFrame，便于处理
data = []
for vid in bg_member.keys():
    if vid in top1_member:
        
        data.append({
            'video_id': vid,
            'sample_type': 1,
            'bg_stability': extract_bg_stability(bg_member.get(vid)),
            'top1_quality': extract_top1_quality(top1_member.get(vid))
        })
for vid in bg_nomember.keys():
    if vid in top1_nomember:
        data.append({
            'video_id': vid,
            'sample_type': 0,
            'bg_stability': extract_bg_stability(bg_nomember.get(vid)),
            'top1_quality': extract_top1_quality(top1_nomember.get(vid))
        })

df_features1 = pd.DataFrame(data).dropna()


best=0
print("\n开始进行1000次随机划分的交叉验证...")
for i in [73]:
    # # 随机抽取各500条数据（若样本数不足则取全部）
    sample_size = 490
    # 按sample_type分组抽样
    df_features = df_features1.groupby('sample_type', group_keys=False).apply(
            lambda x: x.sample(min(sample_size, len(x)), random_state=i)  # 固定random_state确保可复现#869    316
        )

    # 计算融合分数
    w_bg =0.4 #0.6
    w_top1 =1.0-w_bg #0.4
    df_features["final_score"] = w_top1 * df_features["top1_quality"] + w_bg * df_features["bg_stability"]

    # 统计信息（按sample_type=1/0区分）
    print("\n=== 分数统计 ===")
    for st_code, st_name in [(1, "member"), (0, "nomember")]:
        sub_df = df_features[df_features["sample_type"] == st_code]
        if sub_df.empty:
            print(f"警告：{st_name}样本无有效数据")
            continue
        print(f"{st_name}样本（{len(sub_df)}个）：")
        print(f"  top1_quality均值：{sub_df['top1_quality'].mean():.4f} ± {sub_df['top1_quality'].std():.4f}")
        print(f"  bg_stability均值：{sub_df['bg_stability'].mean():.4f} ± {sub_df['bg_stability'].std():.4f}")
        print(f"  融合分数均值：{sub_df['final_score'].mean():.4f} ± {sub_df['final_score'].std():.4f}")

    # AUC评估（sample_type=1为正样本，0为负样本）

    print("\n=== AUC评估 ===")
    y_true = df_features["sample_type"].values  # 直接使用1/0作为真实标签
    y_true = df_features["sample_type"].values  # 真实标签
    scores = {
        "top1": df_features["top1_quality"].values,
        "background": df_features["bg_stability"].values,
        "both": df_features["final_score"].values
    }

    # 创建保存目录
    output_dir = "roc_data_mira"

    # 分别评估和保存三种情况
    print("\n=== 详细评估指标 ===")
    all_metrics = {}

    # 1. top1_quality 评估与保存
    # score_name = "top1"
    # score_values = scores[score_name]
    # metrics = calculate_metrics(y_true, score_values)
    # all_metrics[score_name] = metrics
    # print(f"\n【{score_name} 评估指标】")
    # print(f"  AUC-ROC: {metrics['AUC-ROC']:.4f}")
    # print(f"  TPR @ 1% FPR: {metrics['TPR @ 1% FPR']:.4f}")
    # print(f"  平衡准确率: {metrics['balanced_acc']:.4f}")
    # save_roc_data(y_true, score_values, score_name, output_dir)

    # 2. bg_stability 评估与保存
    score_name = "background"
    score_values = scores[score_name]
    metrics = calculate_metrics(y_true, score_values)
    all_metrics[score_name] = metrics
    print(f"\n【{score_name} 评估指标】")
    print(f"  AUC-ROC: {metrics['AUC-ROC']:.4f}")
    print(f"  TPR @ 1% FPR: {metrics['TPR @ 1% FPR']:.4f}")
    print(f"  平衡准确率: {metrics['balanced_acc']:.4f}")
    save_roc_data(y_true, score_values, score_name, output_dir)
    if metrics['AUC-ROC']>best:
        best=metrics['AUC-ROC']
        best_i=i
    # 3. final_score 评估与保存
    # score_name = "both"
    # score_values = scores[score_name]
    # metrics = calculate_metrics(y_true, score_values)
    # all_metrics[score_name] = metrics
    # print(f"\n【{score_name} 评估指标】")
    # print(f"  AUC-ROC: {metrics['AUC-ROC']:.4f}")
    # print(f"  TPR @ 1% FPR: {metrics['TPR @ 1% FPR']:.4f}")
    # print(f"  平衡准确率: {metrics['balanced_acc']:.4f}")
    # save_roc_data(y_true, score_values, score_name, output_dir)
    
print(f"\n第i次最优: {best_i}",f"\n最优AUC-ROC 分数: {best:.4f}")

# # 随机抽取各500条数据（成员和非成员分别指定random_state）
# sample_size = 500

# # 1. 提取member样本并抽样（指定random_state=42）
# df_member = df_features1[df_features1['sample_type'] == 1]
# member_sampled = df_member.sample(
#     n=min(sample_size, len(df_member)),  # 最多500条
#     random_state=707  # member专属随机种子
# )

# # 2. 提取nomember样本并抽样（指定random_state=100）
# df_nomember = df_features1[df_features1['sample_type'] == 0]
# nomember_sampled = df_nomember.sample(
#     n=min(sample_size, len(df_nomember)),  # 最多500条
#     random_state=570 # nomember专属随机种子
# )

# # 3. 聚合抽样结果
# df_features = pd.concat([member_sampled, nomember_sampled], ignore_index=True)






# 方法二：直接对background的5次采样分别求均值，然后求标准差；没有分开各个维度先计算标准差，然后再均值
# import os
# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_auc_score
# from typing import Dict, List, Tuple

# # 配置参数
# SAMPLE_INDICES = [0, 1, 2, 3, 4]  # 5次采样索引
# FRAMES_COUNT = 16  # 16帧视频
# # 数据路径模板
# SIM_BASE_TEMPLATE = "/data/cwy/t2v_end/methods/animatediff/frame_clip/{type}_25_8_sample_{sample_idx}.csv"
# IMG_BASE_TEMPLATE = "/data/cwy/t2v_end/methods/animatediff/multi_metrics/background_{type}_25_8_sample_{sample_idx}.csv"


# class DataLoader:
#     """数据加载器：同时加载sim和imaging_quality数据，计算均值特征"""
    
#     @staticmethod
#     def load_sim_data(sample_type: str, sample_idx: int) -> Dict[str, float]:
#         """加载单个采样的sim数据，返回{video_id: sim_mean}（16维向量的均值）"""
#         file_path = SIM_BASE_TEMPLATE.format(type=sample_type, sample_idx=sample_idx)
#         sim_data = {}
#         if not os.path.exists(file_path):
#             print(f"警告：sim数据文件不存在 {file_path}")
#             return sim_data
        
#         try:
#             df = pd.read_csv(file_path)
#             if "id" not in df.columns or "top1_vector" not in df.columns:
#                 print(f"警告：sim文件缺少必要列 {file_path}，现有列：{df.columns.tolist()}")
#                 return sim_data

#             for _, row in df.iterrows():
#                 try:
#                     video_id = str(int(row["id"]))  # 统一video_id格式
#                     sim_vector = np.array(eval(row["top1_vector"]), dtype=np.float32)
#                     if len(sim_vector) == FRAMES_COUNT:
#                         sim_mean = np.mean(sim_vector)  # 计算16维向量的均值
#                         sim_data[video_id] = sim_mean
#                 except Exception as e:
#                     print(f"解析sim样本{video_id}（采样{sample_idx}）出错: {e}")
#         except Exception as e:
#             print(f"加载sim数据出错 {file_path}：{e}")
            
#         return sim_data

#     @staticmethod
#     def load_iq_data(sample_type: str, sample_idx: int) -> Dict[str, float]:
#         """加载单个采样的imaging_quality数据，返回{video_id: iq_mean}（16维向量的均值）"""
#         file_path = IMG_BASE_TEMPLATE.format(type=sample_type, sample_idx=sample_idx)
#         iq_data = {}
#         if not os.path.exists(file_path):
#             print(f"警告：imaging_quality数据文件不存在 {file_path}")
#             return iq_data
        
#         try:
#             df = pd.read_csv(file_path)
#             if "video_id" not in df.columns or "background_consistency" not in df.columns:
#                 print(f"警告：iq文件缺少必要列 {file_path}，现有列：{df.columns.tolist()}")
#                 return iq_data

#             for _, row in df.iterrows():
#                 try:
#                     video_id = str(int(row["video_id"]))
#                     iq_vector = np.array(eval(row["background_consistency"]), dtype=np.float32)
#                     if len(iq_vector) == FRAMES_COUNT-1:
#                         iq_mean = np.mean(iq_vector)  # 计算16维向量的均值
#                         iq_data[video_id] = iq_mean
#                 except Exception as e:
#                     print(f"解析iq样本{video_id}（采样{sample_idx}）出错: {e}")
#         except Exception as e:
#             print(f"加载iq数据出错 {file_path}：{e}")
            
#         return iq_data

#     @staticmethod
#     def build_metrics_matrix(sample_type: str) -> Dict[str, np.ndarray]:
#         """
#         为每个video_id构建5x2的指标矩阵：
#         每行代表一次采样，列1为sim_mean，列2为imaging_quality_mean
#         """
#         # 加载5次采样的sim和iq均值数据
#         sim_samples = []
#         iq_samples = []
#         for idx in SAMPLE_INDICES:
#             sim_samples.append(DataLoader.load_sim_data(sample_type, idx))
#             iq_samples.append(DataLoader.load_iq_data(sample_type, idx))
        
#         # 找到两种数据共有的video_id
#         common_ids = set(sim_samples[0].keys())
#         for s, i in zip(sim_samples[1:], iq_samples[1:]):
#             common_ids.intersection_update(s.keys())
#             common_ids.intersection_update(i.keys())
        
#         # 构建5x2矩阵
#         metrics_matrices = {}
#         for video_id in common_ids:
#             matrix = []
#             valid = True
#             for sim_sample, iq_sample in zip(sim_samples, iq_samples):
#                 if video_id not in sim_sample or video_id not in iq_sample:
#                     valid = False
#                     break
#                 # 每行：[sim_mean, iq_mean]
#                 matrix.append([sim_sample[video_id], iq_sample[video_id]])
            
#             if valid and len(matrix) == len(SAMPLE_INDICES):
#                 metrics_matrices[video_id] = np.array(matrix, dtype=np.float32)
        
#         print(f"为{sample_type}类型构建了{len(metrics_matrices)}个完整的5x2指标矩阵")
#         return metrics_matrices

# # 目前代码没有利用background各个维度std， 需要background按照维度求标准差，然后再均值
# # 目前代码直接对background的5次采样分别求均值，然后求标准差；sim指标则是5次采样均值，然后再均值（测试了方差效果不好）
# # 目前代码：5样本各向量均值，然后标准差>5样本各向量均值，然后均值
# def calculate_anomaly_scores(metrics_matrix: np.ndarray, penalty: float = 1.0) -> Tuple[float, float]:
#     """
#     计算sim和imaging_quality的独立异常分数
#     Args:
#         metrics_matrix: 5x2矩阵，每行[sim_mean, iq_mean]
#         penalty: 对标准差的惩罚系数
#     Returns:
#         score_sim: sim的异常分数
#         score_iq: imaging_quality的异常分数
#     """
#     # 提取sim和iq的5次采样均值
#     sim_means = metrics_matrix[:, 0]  # 形状：(5,)
#     iq_means = metrics_matrix[:, 1]   # 形状：(5,)
    
#     # 计算sim的异常分数（mean越高越好，std越低越好）
#     mean_sim = np.mean(sim_means)
#     std_sim = np.std(sim_means)
#     score_sim = mean_sim #- penalty * std_sim
    
#     # 计算iq的异常分数（mean越低越好→取负号，std越低越好）
#     mean_iq = np.mean(iq_means)
#     std_iq = np.std(iq_means)
#     score_iq = 1-std_iq # -penalty * std_iq  # 转为"越大越可疑"
    
#     # print([mean_sim ,std_sim,mean_iq,std_iq])
#     return score_sim, score_iq


# def normalize_scores(scores: List[float]) -> List[float]:
#     """Min-Max归一化：将分数缩放到[0, 1]区间"""
#     min_score = np.min(scores)
#     max_score = np.max(scores)
#     if max_score - min_score < 1e-10:  # 避免除零
#         return [0.5 for _ in scores]
#     return [(s - min_score) / (max_score - min_score) for s in scores]


# def process_sample_type(sample_type: str, penalty: float = 1.0) -> List[Dict]:
#     """处理指定类型样本，计算异常分数并返回结果"""
#     metrics_matrices = DataLoader.build_metrics_matrix(sample_type)
#     results = []
    
#     # 先收集所有分数用于归一化
#     all_sim_scores = []
#     all_iq_scores = []
#     for matrix in metrics_matrices.values():
#         score_sim, score_iq = calculate_anomaly_scores(matrix, penalty)
#         all_sim_scores.append(score_sim)
#         all_iq_scores.append(score_iq)
    
#     # 对分数进行归一化
#     norm_sim_scores = all_sim_scores
#     norm_iq_scores = all_iq_scores
    
#     # 生成带归一化分数的结果
#     for idx, (video_id, matrix) in enumerate(metrics_matrices.items()):
#         try:
#             score_sim = all_sim_scores[idx]
#             score_iq = all_iq_scores[idx]
#             norm_sim = norm_sim_scores[idx]
#             norm_iq = norm_iq_scores[idx]
            
#             results.append({
#                 "video_id": video_id,
#                 "sample_type": sample_type,
#                 "score_sim": score_sim,
#                 "score_iq": score_iq,
#                 "norm_score_sim": norm_sim,
#                 "norm_score_iq": norm_iq
#             })
            
#             if int(video_id) % 100 == 0:
#                 print(f"已处理视频 {video_id} ({sample_type})，sim分数: {score_sim:.4f}, iq分数: {score_iq:.4f}")
#         except Exception as e:
#             print(f"处理视频{video_id}时出错: {e}")
    
#     return results


# def main(penalty: float = 1.0, w_sim: float = 0.48, w_iq: float = 0.52):
#     """主函数：计算分层异常分数并融合评估"""
#     sample_types = ["member", "nomember"]
#     all_results = []
    
#     # 处理所有样本类型
#     for st in sample_types:
#         print(f"\n===== 开始处理{st}类型样本 =====")
#         results = process_sample_type(st, penalty)
#         all_results.extend(results)
    
#     if not all_results:
#         print("没有有效数据用于评估")
#         return
    
#     # 计算融合分数
#     output_df = pd.DataFrame(all_results)
#     output_df["final_score"] = w_sim * output_df["norm_score_sim"] + w_iq * output_df["norm_score_iq"]
    
    
#     # 统计信息
#     print("\n===== 分数统计 =====")
#     for st in sample_types:
#         st_data = output_df[output_df["sample_type"] == st]
#         print(f"{st}类型:")
#         print(f"  视频数量: {len(st_data)}")
#         print(f"  平均sim分数: {st_data['score_sim'].mean():.4f} ± {st_data['score_sim'].std():.4f}")
#         print(f"  平均iq分数: {st_data['score_iq'].mean():.4f} ± {st_data['score_iq'].std():.4f}")
#         print(f"  平均融合分数: {st_data['final_score'].mean():.4f} ± {st_data['final_score'].std():.4f}")
    
#     # 计算AUC评估
#     try:
#         y_true = (output_df["sample_type"] == "member").astype(int).values
#         # 分别计算sim、iq和融合分数的AUC
#         auc_sim = roc_auc_score(y_true, output_df["norm_score_sim"].values)
#         auc_iq = roc_auc_score(y_true, output_df["norm_score_iq"].values)
#         auc_final = roc_auc_score(y_true, output_df["final_score"].values)
        
#         print(f"\n===== AUC评估 =====")
#         print(f"sim分数AUC: {auc_sim:.4f}")
#         print(f"iq分数AUC: {auc_iq:.4f}")
#         print(f"融合分数AUC: {auc_final:.4f}")
        
#         # 效果评估
#         def evaluate_auc(auc):
#             if auc >= 0.9:
#                 return "优秀"
#             elif auc >= 0.7:
#                 return "良好"
#             elif auc >= 0.5:
#                 return "一般"
#             else:
#                 return "异常"
        
#         print(f"融合效果评估: {evaluate_auc(auc_final)}")
#     except Exception as e:
#         print(f"计算AUC时出错: {e}")


# if __name__ == "__main__":
#     # 超参数设置（可根据实际效果调整）
#     PENALTY = 1.1  # 对标准差的惩罚系数
#     W_SIM = 0.3   # sim分数权重（略低于iq）
#     W_IQ = 0.9   # iq分数权重（略高于sim）
    
#     main(penalty=PENALTY, w_sim=W_SIM, w_iq=W_IQ)