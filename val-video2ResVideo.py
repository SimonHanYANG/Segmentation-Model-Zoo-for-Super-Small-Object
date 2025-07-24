'''
Author: SimonHanYANG SimonCK666@163.com
Date: 2025-07-15 15:28:26
LastEditors: SimonHanYANG SimonCK666@163.com
LastEditTime: 2025-07-24 14:18:38
FilePath: /UNeXt_pytorch/val-video2ResVideo.py
Description: video as input, res hovered frames video as output

python val-video2ResVideo.py --video_path /home/simon/UNeXt-pytorch/inputs/MermerTestVideo/mermerTest.avi --name ENet_mermerHeadTail_0710
'''
import argparse
import os
from glob import glob
import time
import shutil
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
from archs import UNext

# 导入FEDASNet
from model_zoo.fedas_net import FEDASNet

# python val.py --name ENet_mermerHeadTail_0710 --video_path path_to_video.mp4

# 定义一个修改后的Dataset类，适用于视频帧
class VideoFrameDataset:
    def __init__(self, num_classes, transform=None, input_channels=3):
        self.num_classes = num_classes
        self.transform = transform
        self.input_channels = input_channels
        
    def process_frame(self, frame):
        """处理单个视频帧"""
        # 创建全黑掩码（因为我们只关心预测结果）
        mask = np.zeros((frame.shape[0], frame.shape[1], self.num_classes), dtype=np.uint8)
            
        if self.transform is not None:
            augmented = self.transform(image=frame, mask=mask)
            frame = augmented['image']
            mask = augmented['mask']
            
        # 转换为模型需要的格式
        frame = frame.astype('float32') / 255
        mask = mask.astype('float32') / 255
        
        frame = frame.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        
        return torch.from_numpy(frame), torch.from_numpy(mask)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--cuda', default="cuda:0",
                        help='cuda device')
    parser.add_argument('--video_path', default=None, required=True,
                        help='path to input video file')
    parser.add_argument('--output_fps', type=float, default=None,
                        help='output video fps (default: same as input)')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = args.cuda

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # 创建模型 - 添加对FEDASNet的支持
    print("=> creating model %s" % config['arch'])
    if config['arch'] == 'FEDASNet':
        model = FEDASNet(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            deep_supervision=config.get('deep_supervision', False)
        )
    else:
        model = archs.__dict__[config['arch']](
            config['num_classes'],
            config['input_channels'],
            config.get('deep_supervision', False)
        )

    model = model.to(device)

    # 加载模型权重
    model.load_state_dict(torch.load('models/%s/model.pth' % config['name'], map_location=device))
    model.eval()

    # 设置输出目录
    output_dir = os.path.join('outputs', config['name'])
    os.makedirs(output_dir, exist_ok=True)

    # 视频输入/输出路径
    video_path = args.video_path
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_segmented.mp4")

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 使用指定的输出FPS或默认使用输入视频的FPS
    output_fps = args.output_fps if args.output_fps else fps

    print(f"视频信息: {frame_width}x{frame_height}, {fps} FPS, {total_frames} 总帧数")
    print(f"输出视频将使用 {output_fps} FPS")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))

    # 创建转换
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 创建数据集
    dataset = VideoFrameDataset(
        num_classes=config['num_classes'],
        transform=val_transform,
        input_channels=config['input_channels']
    )

    # 定义叠加颜色（BGR格式）
    overlay_colors = [
        (0, 0, 255),    # 类别0: 红色
        (0, 255, 0),    # 类别1: 绿色
        (255, 0, 0),    # 类别2: 蓝色
        (0, 255, 255)   # 类别3: 黄色
    ]
    opacity = 0.35  # 15%不透明度

    # 评估指标
    processing_times = []  # 存储每帧的处理时间
    is_first_frame = True  # 标记是否是第一帧（用于预热）

    # 处理视频
    print(f"开始处理视频，共 {total_frames} 帧...")
    
    with torch.no_grad():
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存原始帧以便后续叠加
                original_frame = frame.copy()
                
                # 处理帧
                input_tensor, mask_tensor = dataset.process_frame(frame)
                input_tensor = input_tensor.unsqueeze(0).to(device)  # 添加批次维度
                
                # 开始计时
                start_time = time.time()
                
                # 前向传播
                output = model(input_tensor)
                
                # 处理深度监督的情况
                if isinstance(output, list):
                    main_output = output[-1]
                else:
                    main_output = output
                
                # 后处理
                output_prob = torch.sigmoid(main_output).cpu().numpy()
                output_binary = output_prob.copy()
                output_binary[output_binary >= 0.5] = 1
                output_binary[output_binary < 0.5] = 0
                
                # 计算处理时间
                frame_processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
                
                # 如果不是第一帧（预热帧），则记录处理时间
                if not is_first_frame:
                    processing_times.append(frame_processing_time)
                else:
                    is_first_frame = False
                
                # 将预测结果调整回原始图像大小
                for c in range(config['num_classes']):
                    # 调整预测结果大小到原始帧尺寸
                    mask_resized = cv2.resize(
                        output_binary[0, c], 
                        (frame_width, frame_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    # 创建颜色叠加层
                    overlay = np.zeros_like(original_frame)
                    overlay[mask_resized > 0.5] = overlay_colors[c]
                    
                    # 将预测结果叠加到原始帧上
                    cv2.addWeighted(overlay, opacity, original_frame, 1, 0, original_frame)
                
                # 写入输出视频
                out.write(original_frame)
                
                # 更新进度条
                pbar.update(1)
    
    # 释放资源
    cap.release()
    out.release()

    # 计算平均处理时间
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    total_processed_frames = len(processing_times)

    # 打印最终结果
    print('='*50)
    print('Final Results:')
    print('Model: %s' % config['arch'])
    print('Video: %s' % video_path)
    print('Output Video: %s' % output_video_path)
    print('Average Processing Time: %.4f ms per frame (excluding warmup)' % avg_processing_time)
    print('Total Frames Processed (excluding warmup): %d' % total_processed_frames)
    print('='*50)
    
    # 保存评估结果到文件
    result_file = os.path.join(output_dir, f"{video_name}_processing_results.txt")
    with open(result_file, 'w') as f:
        f.write('Model: %s\n' % config['arch'])
        f.write('Video: %s\n' % video_path)
        f.write('Output Video: %s\n' % output_video_path)
        f.write('Original FPS: %.2f\n' % fps)
        f.write('Output FPS: %.2f\n' % output_fps)
        f.write('Frame Size: %dx%d\n' % (frame_width, frame_height))
        f.write('Average Processing Time: %.4f ms per frame (excluding warmup)\n' % avg_processing_time)
        f.write('Total Frames Processed (excluding warmup): %d\n' % total_processed_frames)
        f.write('Total Processing Time: %.4f ms\n' % sum(processing_times))
    
    # 复制配置文件到输出目录，便于追踪实验
    try:
        config_src = os.path.join('models', config['name'], 'config.yml')
        config_dst = os.path.join(output_dir, 'config.yml')
        shutil.copy2(config_src, config_dst)
        print(f"已复制配置文件到: {config_dst}")
    except Exception as e:
        print(f"复制配置文件时出错: {e}")

    torch.cuda.empty_cache()
    
    print(f"分割视频已保存到: {output_video_path}")
    print(f"处理结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
