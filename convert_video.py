import ffmpeg
import os

def convert_to_h264(input_path, output_path):
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 设置转换参数
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path,
                             vcodec='libx264',    # H.264编码
                             acodec='aac',        # AAC音频编码
                             video_bitrate='2000k',# 视频比特率
                             audio_bitrate='128k', # 音频比特率
                             preset='medium',      # 编码速度预设
                             crf=23)              # 质量因子(0-51,越低质量越好)
        
        # 执行转换
        ffmpeg.run(stream, overwrite_output=True)
        print(f"转换成功: {output_path}")
        
    except ffmpeg.Error as e:
        print(f"转换失败: {str(e)}")
        return False
    
    return True

# 使用示例
if __name__ == "__main__":
    # 设置输入输出路径
    input_dir = "static/videos/classify"
    output_dir = "static/videos/h264"
    
    # 遍历所有场景和视频
    for scene in range(1, 7):
        for video in range(1, 6):
            input_path = f"{input_dir}/scene{scene}/{video}.mp4"
            output_path = f"{output_dir}/scene{scene}/{video}.mp4"
            
            if os.path.exists(input_path):
                print(f"正在转换: {input_path}")
                convert_to_h264(input_path, output_path)