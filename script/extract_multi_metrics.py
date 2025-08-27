import json
import csv
import os
from collections import defaultdict

def extract_metrics(json_file, metric_name):
    """从JSON文件中提取指定指标"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 提取视频结果列表
        if metric_name in data and len(data[metric_name]) >= 2:
            return data[metric_name][1]  # 第二个元素是视频结果列表
    except FileNotFoundError:
        print(f"警告: 文件 {json_file} 未找到，将跳过该文件")
    except Exception as e:
        print(f"处理文件 {json_file} 时出错: {str(e)}")
    return []

def get_video_id(video_path):
    """从视频路径中提取视频ID，只取数字部分"""
    filename = os.path.splitext(os.path.basename(video_path))[0]
    # 提取文件名开头的数字部分
    for i, char in enumerate(filename):
        if not char.isdigit():
            return int(filename[:i])
    return int(filename)  # 如果全是数字则返回整个文件名

def process_metrics(types, steps, metrics, sample_inds, output_dir):
    """处理指定类型和步骤的指标并生成CSV文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
        # 增加对sample_inds的循环，遍历所有样本索引
    for sample_ind in sample_inds:
        # 对于每种类型和步骤的组合，处理并生成CSV
        for step in steps:
            for type_ in types:
                # 收集该类型和步骤的所有指标数据
                video_metrics = defaultdict(dict)
                
                # 提取每个指标的JSON文件数据
                for metric_name in metrics:
                    # 构建JSON文件路径
                    # json_path = f"/data/cwy/t2v_end/origin_methods/animatediff/multi_metrics/{metric_name}_{type_}_{step}_8_sample_{sample_ind}/result_eval_results.json"
                    json_path = f"/data/cwy/t2v_end/methods/mira/multi_metrics/{metric_name}_{type_}_sample_{sample_ind}/result_eval_results.json"
                    results = extract_metrics(json_path, metric_name)
                    
                    for item in results:
                        video_path = item["video_path"]
                        video_id = get_video_id(video_path)
                        # 确保只处理当前步骤的视频
                        if f"sample_{sample_ind}" in video_path:
                            video_metrics[video_id]["video_id"] = video_id
                            video_metrics[video_id][metric_name] = item["value_list"]
                
                # 定义CSV表头 - 共5列
                fieldnames = ["video_id"] + metrics
                
                # 生成CSV文件
                if video_metrics:  # 只在有数据时生成文件
                    filename = f"{metric_name}_{type_}_new_sample_{sample_ind}.csv"
                    file_path = os.path.join(output_dir, filename)
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for item in video_metrics.values():
                            # 确保所有字段都存在
                            row = {field: item.get(field, "") for field in fieldnames}
                            writer.writerow(row)
                    print(f"已生成文件: {file_path}")
                else:
                    print(f"没有找到 {type_} {step} 类型的视频指标数据，跳过生成文件")

if __name__ == "__main__":
    # 定义类型和步骤
    TYPES = ["member"]
    STEPS = [25]
    SAMPLE_INDICES = [0,1,2,3]
    
    # 定义要提取的指标
    METRICS = [
        # "aesthetic_quality",
        "background_consistency",
        # "imaging_quality",
        # "subject_consistency",
        # "dynamic_degree"
    ]
    
    # 输出目录
    output_directory = "/data/cwy/t2v_end/methods/mira/multi_metrics"
    
    # 处理并生成CSV文件
    process_metrics(TYPES, STEPS, METRICS, SAMPLE_INDICES, output_directory)
    