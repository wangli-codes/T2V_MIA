import os
import csv
import torch
import clip
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import math

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class FrameSimilarityCalculator:
    def __init__(self, original_dir, generated_dir, output_csv, clip_model="ViT-B/32"):
        self.original_dir = Path(original_dir)
        self.generated_dir = Path(generated_dir)
        self.output_csv = output_csv
        self.clip_model = clip_model
        
        if not self.original_dir.exists():
            raise FileNotFoundError(f"原始视频帧目录不存在: {original_dir}")
        if not self.generated_dir.exists():
            raise FileNotFoundError(f"生成视频帧目录不存在: {generated_dir}")
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)
            print(f"成功加载CLIP模型: {self.clip_model}，运行在{self.device}上")
        except Exception as e:
            raise RuntimeError(f"无法加载CLIP模型: {e}")
    
    def compute_similarity(self, image_path1, image_path2):
        try:
            img1 = Image.open(image_path1)
            image_input1 = self.preprocess(img1).unsqueeze(0).to(self.device)
            
            img2 = Image.open(image_path2)
            image_input2 = self.preprocess(img2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features1 = self.model.encode_image(image_input1)
                image_features2 = self.model.encode_image(image_input2)
            
            similarity = torch.nn.functional.cosine_similarity(image_features1, image_features2)
            return similarity.item()
        except Exception as e:
            print(f"计算相似度失败：{image_path1} 与 {image_path2}，错误：{e}")
            return None
    
    def get_folder_id_origin(self, folder_name):
        try:
            return folder_name.split('_')[0]
            # return str(int(folder_name.split('-')[0]))
        except:
            print(f"警告: 无法从文件夹名 {folder_name} 中提取ID")
            return None
    
    def get_folder_id_generate(self, folder_name):
        try:
            return folder_name.split('-')[0]
            # return str(int(folder_name.split('_')[0]))
            # return folder_name.split('_')[1]
        except:
            print(f"警告: 无法从文件夹名 {folder_name} 中提取ID")
            return None
    
    def calculate_similarity_scores(self):
        results = defaultdict(dict)
        
        id_to_generated = {}
        for folder in self.generated_dir.iterdir():
            if folder.is_dir():
                folder_id = self.get_folder_id_generate(folder.name)
                if folder_id is not None:
                    id_to_generated[folder_id] = folder
        
        for folder in tqdm(self.original_dir.iterdir(), desc="处理原始文件夹"):
            if not folder.is_dir():
                continue
                
            folder_id = self.get_folder_id_origin(folder.name)
            if folder_id is None:
                continue
                
            if folder_id not in id_to_generated:
                print(f"警告: 找不到ID为 {folder_id} 的生成文件夹，跳过")
                continue
                
            generated_folder = id_to_generated[folder_id]
            
            generated_frames = sorted([f for f in generated_folder.iterdir() if f.is_file() and self.is_image_file(f.name)])
            original_frames = sorted([f for f in folder.iterdir() if f.is_file() and self.is_image_file(f.name)])
            if not generated_frames or not original_frames:
                print(f"警告: {folder_id} 中的生成帧或原始帧为空，跳过")
                continue
            
            # 存储每个生成帧的计算结果向量
            frame_average_similarities = []  # 每个生成帧与所有原始帧的平均相似度
            top1_vector = []  # 每个生成帧与原始帧的top1相似度
            top3_vector = []  # 每个生成帧与原始帧的top3平均相似度
            top5_vector = []  # 每个生成帧与原始帧的top5平均相似度
            
            id_results = []
            for gen_frame in generated_frames:
                # 计算当前生成帧与所有原始帧的相似度
                frame_similarities = []
                for orig_frame in original_frames:
                    sim = self.compute_similarity(gen_frame, orig_frame)
                    if sim is not None:
                        frame_similarities.append(sim)
                
                if frame_similarities:
                    # 对当前生成帧的所有原始帧相似度排序（从高到低）
                    sorted_sims = sorted(frame_similarities, reverse=True)
                    
                    # 计算当前生成帧的top1：最高的1个相似度
                    top1 = sorted_sims[0] if len(sorted_sims) >= 1 else math.nan
                    
                    # 计算当前生成帧的top3：最高3个的平均（不足3个则取全部）
                    top3_count = min(3, len(sorted_sims))
                    top3 = sum(sorted_sims[:top3_count]) / top3_count
                    
                    # 计算当前生成帧的top5：最高5个的平均（不足5个则取全部）
                    top5_count = min(5, len(sorted_sims))
                    top5 = sum(sorted_sims[:top5_count]) / top5_count
                    
                    # 计算当前生成帧与所有原始帧的平均相似度
                    frame_avg = sum(frame_similarities) / len(frame_similarities)
                else:
                    # 没有有效相似度时用NaN填充
                    top1 = top3 = top5 = frame_avg = math.nan
                
                # 存储当前生成帧的结果
                id_results.append({
                    'generated_frame': gen_frame.name,
                    'frame_average': frame_avg,
                    'top1': top1,
                    'top3': top3,
                    'top5': top5
                })
                
                # 添加到向量中
                frame_average_similarities.append(frame_avg)
                top1_vector.append(top1)
                top3_vector.append(top3)
                top5_vector.append(top5)
            
            if id_results:
                # 过滤NaN值，仅保留有效相似度
                valid_averages = [v for v in frame_average_similarities if not math.isnan(v)]
                if not valid_averages:
                    print(f"警告: {folder_id} 中没有有效相似度数据，跳过")
                    continue
                
                avg_cross_similarity = sum(valid_averages) / len(valid_averages)
                
                results[folder_id] = {
                    'num_generated_frames': len(generated_frames),
                    'num_original_frames': len(original_frames),
                    'avg_cross_similarity': avg_cross_similarity,  # 所有生成帧平均相似度的平均值
                    'cross_similarities_vector': frame_average_similarities,  # 每个生成帧的平均相似度向量
                    'top1_vector': top1_vector,  # 每个生成帧的top1相似度向量
                    'top3_vector': top3_vector,  # 每个生成帧的top3平均相似度向量
                    'top5_vector': top5_vector,  # 每个生成帧的top5平均相似度向量
                    'frame_results': id_results
                }
        
        return results
    
    def is_image_file(self, filename):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        return any(filename.lower().endswith(ext) for ext in img_extensions)
    
    def save_to_csv(self, results):
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'num_generated_frames', 'num_original_frames', 
                         'avg_cross_similarity', 'cross_similarities_vector',
                         'top1_vector', 'top3_vector', 'top5_vector']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for id_name, data in results.items():
                # 将向量转换为字符串存储
                similarity_vector_str = ",".join(map(str, data['cross_similarities_vector']))
                top1_vector_str = ",".join(map(str, data['top1_vector']))
                top3_vector_str = ",".join(map(str, data['top3_vector']))
                top5_vector_str = ",".join(map(str, data['top5_vector']))
                
                writer.writerow({
                    'id': id_name,
                    'num_generated_frames': data['num_generated_frames'],
                    'num_original_frames': data['num_original_frames'],
                    'avg_cross_similarity': data['avg_cross_similarity'],
                    'cross_similarities_vector': similarity_vector_str,
                    'top1_vector': top1_vector_str,
                    'top3_vector': top3_vector_str,
                    'top5_vector': top5_vector_str
                })
        
        print(f"结果已保存到 {self.output_csv}")
        print(f"当前计算逻辑：")
        print(f"- cross_similarities_vector：每个生成帧与所有原始帧的平均相似度（长度=生成帧数量）")
        print(f"- top1_vector：每个生成帧与原始帧的最高相似度（长度=生成帧数量）")
        print(f"- top3_vector：每个生成帧与原始帧的最高3个相似度的平均值（不足3个则取全部，长度=生成帧数量）")
        print(f"- top5_vector：每个生成帧与原始帧的最高5个相似度的平均值（不足5个则取全部，长度=生成帧数量）")
        print(f"- avg_cross_similarity：cross_similarities_vector中所有有效值的平均值")
    
    def run(self):
        print(f"开始计算视频帧相似度，使用 CLIP ({self.clip_model}) 特征提取方法")
        results = self.calculate_similarity_scores()
        self.save_to_csv(results)
        print("相似度计算完成")

def main():
    param1 = "25"
    param2 = "0"

    # original_dir = "/data/cwy/t2v_end/datasets/panda-70M/download_sample_250_compress_keyframe"
    # generated_dir = "/data/cwy/t2v_end/methods/mira/allframes/nomember_sample_0"
    # output_csv = "/data/cwy/t2v_end/methods/mira/frame_clip/nomember_sample_0.csv"

    # original_dir = "/data/cwy/t2v_end/datasets/mira/download_sample_250_keyframe"
    # generated_dir = "/data/cwy/t2v_end/methods/mira/allframes/member_sample_0"
    # output_csv = "/data/cwy/t2v_end/methods/mira/frame_clip/test_member_sample_0.csv"


    original_dir = "/data/cwy/t2v_end/datasets/panda-70M/download_sample_600_keyframe_two"
    generated_dir = f"/data/cwy/t2v_end/methods/animatediff/allframes/nomember_{param1}_8_sample_{param2}"
    output_csv = f"/data/cwy/t2v_end/methods/animatediff/frame_clip/nomember_{param1}_8_sample_{param2}_four.csv"


    # original_dir = "/data/cwy/t2v_end/datasets/webvid-10M/webvid_download_sample_600_keyframe_three"
    # generated_dir = f"/data/cwy/t2v_end/methods/animatediff/allframes/member_{param1}_8_sample_{param2}"
    # output_csv = f"/data/cwy/t2v_end/methods/animatediff/frame_clip/member_{param1}_8_sample_{param2}_four.csv"

    # original_dir = "/data/cwy/t2v_end/datasets/panda-70M/download_sample_600_keyframe"
    # generated_dir = f"/data/cwy/t2v_end/methods/instructVideo/allframes/nomember_{param1}_0.1_sample_{param2}"
    # output_csv = f"/data/cwy/t2v_end/methods/instructVideo/frame_clip/nomember_{param1}_0.1_sample_{param2}.csv"


    # original_dir = "/data/cwy/t2v_end/datasets/instruct-webvid/instruct_webvid_download_sample_423_keyframe"
    # generated_dir = f"/data/cwy/t2v_end/methods/instructVideo/allframes/member_{param1}_0.1_sample_{param2}"
    # output_csv = f"/data/cwy/t2v_end/methods/instructVideo/frame_clip/member_{param1}_0.1_sample_{param2}.csv"


    calculator = FrameSimilarityCalculator(original_dir, generated_dir, output_csv)
    calculator.run()

if __name__ == "__main__":
    main()
