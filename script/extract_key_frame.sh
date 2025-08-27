#!/bin/bash

# 视频文件所在目录
VIDEO_DIR="/data/cwy/t2v_end/datasets/mira/download_sample_250_new"

# 输出基础目录
OUTPUT_BASE="/data/cwy/t2v_end/datasets/mira/download_sample_250_new_keyframe"

# 检查视频目录是否存在
if [ ! -d "$VIDEO_DIR" ]; then
    echo "错误: 视频目录 '$VIDEO_DIR' 不存在!"
    exit 1
fi

# 确保输出基础目录存在
mkdir -p "$OUTPUT_BASE"

# 遍历视频目录中的所有文件
for video_file in "$VIDEO_DIR"/*; do
    # 检查是否为文件
    if [ -f "$video_file" ]; then
        # 获取文件名（不含路径）
        filename=$(basename "$video_file")
        
        # 获取文件名（不含扩展名）
        filename_noext="${filename%.*}"
        
        # 为当前视频创建输出目录
        output_dir="$OUTPUT_BASE/$filename_noext"
        mkdir -p "$output_dir"
        
        # 输出处理信息
        echo "正在处理视频: $filename"
        echo "输出目录: $output_dir"
        
        # 执行ffmpeg命令提取关键帧
        ffmpeg -i "$video_file" -vf "select='eq(pict_type\,I)'" -vsync 2 -f image2 "$output_dir/keyframe-%04d.jpeg"
        # ffmpeg -i "$video_file" -vsync 0 "$output_dir/frame-%04d.jpeg"
        
        # 检查ffmpeg命令是否成功
        if [ $? -eq 0 ]; then
            echo "成功从 $filename 提取关键帧"
        else
            echo "错误: 从 $filename 提取关键帧失败"
        fi
        echo
    fi
done

echo "所有视频处理完成!"    