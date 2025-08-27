import os
import csv
import time
import re
from google import genai
from tqdm import tqdm

client = genai.Client(api_key='xxxx')    
video_folder = "/data/cwy/t2v_end/datasets/mira/download_sample_250_new_compress"
csv_file = "/data/cwy/t2v_end/datasets/mira/download_sample_250_new_caption.csv"

def extract_id_from_filename(filename):
    fileid = filename.split('-')[0]
    # fileid = filename.split('.')[0]
    return str(fileid)

def get_existing_ids(csv_file):
    """获取CSV文件中已存在的ID列表"""
    existing_ids = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            # 跳过标题行
            next(reader, None)
            for row in reader:
                if row:
                    existing_ids.add(row[0])
    return existing_ids

def process_video_file(video_path):
    """处理单个视频文件并返回摘要"""
    try:
        print(f"处理视频: {video_path}")
        # 上传视频文件
        myfile = client.files.upload(file=video_path)
        # 等待文件上传完成
        time.sleep(8)
        
        # 生成内容摘要
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=[myfile, "Please faithfully summarize the following video in one sentence."]
        )
        
        print(f"摘要: {response.text}")
        return response.text
    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {str(e)}")
        return None

def main():
    # 获取已处理的ID列表
    existing_ids = get_existing_ids(csv_file)
    
    # 获取所有视频文件
    video_files = []
    for filename in os.listdir(video_folder):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(filename)
    
    # 计算需要处理的文件数量
    total_files = len(video_files)
    files_to_process = [f for f in video_files if extract_id_from_filename(f) not in existing_ids]
    processed_count = total_files - len(files_to_process)
    print(files_to_process)
    print(f"找到 {total_files} 个视频文件，其中 {len(files_to_process)} 个需要处理")
    
    # 检查CSV文件是否存在，如果不存在则创建并写入标题行
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['id', 'filename', 'summary'])
    
    # 使用tqdm显示进度条
    with tqdm(total=total_files, initial=processed_count, desc="处理进度", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        # 遍历视频文件夹
        for filename in video_files:
            # 检查文件是否为视频文件
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                file_id = extract_id_from_filename(filename)
                if not file_id:
                    print(f"无法从文件名 {filename} 中提取ID，跳过")
                    pbar.update(1)
                    continue
                
                # 检查ID是否已处理
                if file_id in existing_ids:
                    pbar.set_postfix_str(f"跳过: {filename}")
                    pbar.update(1)
                    continue
                
                # 处理视频文件
                video_path = os.path.join(video_folder, filename)
                pbar.set_postfix_str(f"正在处理: {filename}")
                
                summary = process_video_file(video_path)
                if summary:
                    # 将结果写入CSV
                    with open(csv_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([file_id, filename, summary])
                    pbar.set_postfix_str(f"已保存: {file_id}")
                else:
                    pbar.set_postfix_str(f"处理失败: {file_id}")
                
                pbar.update(1)
                
                # 避免频繁请求API
                time.sleep(2)
    
    print("处理完成!")

if __name__ == "__main__":
    main()
