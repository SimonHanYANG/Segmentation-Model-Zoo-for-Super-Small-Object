'''
Author: SimonHanYANG SimonCK666@163.com
Date: 2025-07-15 15:28:26
LastEditors: SimonHanYANG SimonCK666@163.com
LastEditTime: 2025-07-16 16:25:18
Description: 两阶段精子检测和分类

用法:
# python val-ROIVideo2VideoRes.py --video_path /path/to/video.mp4 --head_model XY-local-UNeXt-all299-0804 --roi_model ENet_sperm_ROINAHead_250707 --output_fps 30 --roi_batch_size 8
python val-ROIVideo2VideoRes.py --video_path '/home/simon/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_115426118.mp4' --head_model XY-local-UNeXt-all299-0804 --roi_model ENet_sperm_ROINAHead_250707 --roi_batch_size 8
python val-ROIVideo2VideoRes.py --video_path '/home/simon/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_115540765.mp4' --head_model XY-local-UNeXt-all299-0804 --roi_model ENet_sperm_ROINAHead_250707 --roi_batch_size 8
python val-ROIVideo2VideoRes.py --video_path '/home/simon/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_115725389.mp4' --head_model XY-local-UNeXt-all299-0804 --roi_model ENet_sperm_ROINAHead_250707 --roi_batch_size 8
python val-ROIVideo2VideoRes.py --video_path '/home/simon/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_120006677.mp4' --head_model XY-local-UNeXt-all299-0804 --roi_model ENet_sperm_ROINAHead_250707 --roi_batch_size 8
python val-ROIVideo2VideoRes.py --video_path '/home/simon/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_121044198.mp4' --head_model XY-local-UNeXt-all299-0804 --roi_model ENet_sperm_ROINAHead_250707 --roi_batch_size 8
python val-ROIVideo2VideoRes.py --video_path '/home/simon/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_121237814.mp4' --head_model XY-local-UNeXt-all299-0804 --roi_model ENet_sperm_ROINAHead_250707 --roi_batch_size 8



'''
import argparse
import os
import time
import shutil
import numpy as np
from collections import deque

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

import archs
from archs import UNext
from model_zoo.fedas_net import FEDASNet
from albumentations import RandomRotate90, Resize

# 定义用于视频帧的Dataset类
class VideoFrameDataset:
    def __init__(self, num_classes, transform=None, input_channels=3, is_grayscale=False):
        self.num_classes = num_classes
        self.transform = transform
        self.input_channels = input_channels
        self.is_grayscale = is_grayscale
        
    def process_frame(self, frame):
        """处理单个视频帧"""
        # 如果需要灰度处理
        if self.is_grayscale:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.input_channels == 3:
                # 灰度图像复制成三通道
                frame = np.stack([gray_frame] * 3, axis=2)
            else:
                frame = gray_frame.reshape(gray_frame.shape[0], gray_frame.shape[1], 1)
        
        # 创建全黑掩码
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

# 提取预测中的精子头部ROI区域
def extract_head_rois(binary_mask, original_frame, roi_size=64):
    """
    从二值掩码中提取精子头部ROI
    
    Args:
        binary_mask: 头部预测的二值掩码
        original_frame: 原始视频帧
        roi_size: ROI大小
        
    Returns:
        rois: 提取的ROIs列表
        roi_positions: ROI在原始帧中的位置 (x, y, w, h)
    """
    # 寻找连通区域
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    roi_positions = []
    
    for contour in contours:
        # 计算轮廓的矩形边界
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        # 计算质心
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 计算ROI的边界，确保在图像内
        half_size = roi_size // 2
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(original_frame.shape[1], cx + half_size)
        y2 = min(original_frame.shape[0], cy + half_size)
        
        # 如果ROI太小，则跳过
        if x2 - x1 < roi_size / 2 or y2 - y1 < roi_size / 2:
            continue
        
        # 提取ROI
        roi = original_frame[y1:y2, x1:x2]
        
        # 如果ROI不是正方形，则进行填充
        if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
            # 创建空白ROI
            padded_roi = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
            # 将提取的ROI放入中心
            h, w = roi.shape[:2]
            pad_y = (roi_size - h) // 2
            pad_x = (roi_size - w) // 2
            padded_roi[pad_y:pad_y+h, pad_x:pad_x+w] = roi
            roi = padded_roi
        
        rois.append(roi)
        roi_positions.append((x1, y1, x2-x1, y2-y1))
    
    return rois, roi_positions

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--head_model', default='XY-local-UNeXt-all299-0804',
                        help='精子头部检测模型名称')
    parser.add_argument('--roi_model', default='ENet_sperm_ROINAHead_250707',
                        help='ROI分类模型名称')
    parser.add_argument('--cuda', default="cuda:0",
                        help='cuda device')
    parser.add_argument('--video_path', default=None, required=True,
                        help='输入视频路径')
    parser.add_argument('--output_fps', type=float, default=None,
                        help='输出视频帧率 (默认：与输入相同)')
    parser.add_argument('--roi_size', type=int, default=64,
                        help='ROI区域大小')
    parser.add_argument('--roi_batch_size', type=int, default=8,
                        help='ROI预测批处理大小')

    args = parser.parse_args()

    return args

def load_model(model_name, device):
    """加载模型及其配置"""
    with open(f'models/{model_name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型
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
    model.load_state_dict(torch.load(f'models/{model_name}/model.pth', map_location=device))
    model.eval()
    
    return model, config

def main():
    args = parse_args()
    device = args.cuda
    cudnn.benchmark = True
    
    # 创建输出目录
    output_dir = os.path.join('outputs', f"{args.head_model}_{args.roi_model}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载精子头部检测模型
    print(f"加载头部检测模型: {args.head_model}")
    head_model, head_config = load_model(args.head_model, device)
    
    # 加载ROI分类模型
    print(f"加载ROI分类模型: {args.roi_model}")
    roi_model, roi_config = load_model(args.roi_model, device)
    
    # 视频输入/输出路径
    video_path = args.video_path
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_analyzed.mp4")
    
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
    
    # 创建头部检测转换
    head_transform = Compose([
        Resize(head_config['input_h'], head_config['input_w']),
        transforms.Normalize(),
    ])
    
    # 创建ROI分类转换
    roi_transform = Compose([
        Resize(roi_config['input_h'], roi_config['input_w']),
        transforms.Normalize(),
    ])
    
    # 创建数据集
    head_dataset = VideoFrameDataset(
        num_classes=head_config['num_classes'],
        transform=head_transform,
        input_channels=head_config['input_channels'],
        is_grayscale=True  # 头部检测使用灰度图像
    )
    
    roi_dataset = VideoFrameDataset(
        num_classes=roi_config['num_classes'],
        transform=roi_transform,
        input_channels=roi_config['input_channels']
    )
    
    # 定义叠加颜色（BGR格式）
    roi_overlay_colors = [
        (0, 0, 255),    # 类别0: 红色
        (0, 255, 0),    # 类别1: 绿色
        (255, 0, 0),    # 类别2: 蓝色
        (0, 255, 255)   # 类别3: 黄色
    ]
    opacity = 0.25  # 15%不透明度
    roi_box_color = (0, 255, 0)  # 绿色框
    
    # 处理指标
    head_processing_times = []
    roi_processing_times = []
    rois_per_frame = []
    is_first_frame = True
    
    # 处理视频
    print(f"开始处理视频，共 {total_frames} 帧...")
    
    with torch.no_grad():
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存原始帧
                original_frame = frame.copy()
                
                # 第一阶段：头部检测
                head_start_time = time.time()
                
                # 处理帧
                input_tensor, _ = head_dataset.process_frame(frame)
                input_tensor = input_tensor.unsqueeze(0).to(device)  # 添加批次维度
                
                # 头部检测
                head_output = head_model(input_tensor)
                
                # 处理深度监督的情况
                if isinstance(head_output, list):
                    head_output = head_output[-1]
                
                # 后处理
                head_prob = torch.sigmoid(head_output).cpu().numpy()
                head_binary = head_prob.copy()
                head_binary[head_binary >= 0.5] = 1
                head_binary[head_binary < 0.5] = 0
                
                # 将二值掩码调整回原始大小
                head_mask_resized = cv2.resize(
                    head_binary[0, 0], 
                    (frame_width, frame_height), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                head_end_time = time.time()
                head_processing_time = (head_end_time - head_start_time) * 1000  # 转换为毫秒
                
                # 第二阶段：提取ROI并分类
                roi_start_time = time.time()
                
                # 提取头部ROI
                rois, roi_positions = extract_head_rois(
                    head_mask_resized, 
                    original_frame, 
                    roi_size=args.roi_size
                )
                
                rois_per_frame.append(len(rois))
                
                # 批量处理ROI
                roi_results = []
                
                # 使用批处理进行ROI分类
                for i in range(0, len(rois), args.roi_batch_size):
                    batch_rois = rois[i:i+args.roi_batch_size]
                    if not batch_rois:
                        continue
                    
                    # 处理批次中的每个ROI
                    batch_tensors = []
                    for roi in batch_rois:
                        roi_tensor, _ = roi_dataset.process_frame(roi)
                        batch_tensors.append(roi_tensor)
                    
                    # 如果批次为空，跳过
                    if not batch_tensors:
                        continue
                    
                    # 创建批次张量
                    batch_input = torch.stack(batch_tensors).to(device)
                    
                    # ROI分类
                    roi_output = roi_model(batch_input)
                    
                    # 处理深度监督的情况
                    if isinstance(roi_output, list):
                        roi_output = roi_output[-1]
                    
                    # 后处理
                    roi_prob = torch.sigmoid(roi_output).cpu().numpy()
                    roi_binary = roi_prob.copy()
                    roi_binary[roi_binary >= 0.5] = 1
                    roi_binary[roi_binary < 0.5] = 0
                    
                    roi_results.extend([roi_binary[j] for j in range(len(batch_rois))])
                
                roi_end_time = time.time()
                roi_processing_time = (roi_end_time - roi_start_time) * 1000  # 转换为毫秒
                
                # 如果不是第一帧（预热帧），则记录处理时间
                if not is_first_frame:
                    head_processing_times.append(head_processing_time)
                    roi_processing_times.append(roi_processing_time)
                else:
                    is_first_frame = False
                
                # 可视化结果
                # 1. 绘制ROI框和分类结果
                for idx, ((x, y, w, h), roi_binary) in enumerate(zip(roi_positions, roi_results)):
                    # 绘制ROI框
                    cv2.rectangle(original_frame, (x, y), (x+w, y+h), roi_box_color, 1)
                    
                    # 将ROI分类结果缩放到对应大小并叠加到原始图像
                    for c in range(roi_config['num_classes']):
                        # 创建掩码
                        roi_mask = np.zeros((h, w), dtype=np.uint8)
                        class_mask_resized = cv2.resize(
                            roi_binary[c], 
                            (w, h), 
                            interpolation=cv2.INTER_NEAREST
                        )
                        roi_mask[class_mask_resized > 0.5] = 1
                        
                        # 创建颜色叠加层
                        overlay = np.zeros_like(original_frame)
                        overlay[y:y+h, x:x+w][roi_mask > 0] = roi_overlay_colors[c]
                        
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
    avg_head_time = sum(head_processing_times) / len(head_processing_times) if head_processing_times else 0
    avg_roi_time = sum(roi_processing_times) / len(roi_processing_times) if roi_processing_times else 0
    avg_rois = sum(rois_per_frame) / len(rois_per_frame) if rois_per_frame else 0
    total_processed_frames = len(head_processing_times)
    
    # 打印最终结果
    print('='*50)
    print('最终结果:')
    print('头部检测模型: %s' % args.head_model)
    print('ROI分类模型: %s' % args.roi_model)
    print('视频: %s' % video_path)
    print('输出视频: %s' % output_video_path)
    print('头部检测平均处理时间: %.4f ms/帧' % avg_head_time)
    print('ROI分类平均处理时间: %.4f ms/帧' % avg_roi_time)
    print('每帧平均ROI数量: %.2f' % avg_rois)
    print('总处理帧数 (不含预热): %d' % total_processed_frames)
    print('='*50)
    
    # 保存评估结果到文件
    result_file = os.path.join(output_dir, f"{video_name}_processing_results.txt")
    with open(result_file, 'w') as f:
        f.write('头部检测模型: %s\n' % args.head_model)
        f.write('ROI分类模型: %s\n' % args.roi_model)
        f.write('视频: %s\n' % video_path)
        f.write('输出视频: %s\n' % output_video_path)
        f.write('原始FPS: %.2f\n' % fps)
        f.write('输出FPS: %.2f\n' % output_fps)
        f.write('帧大小: %dx%d\n' % (frame_width, frame_height))
        f.write('头部检测平均处理时间: %.4f ms/帧\n' % avg_head_time)
        f.write('ROI分类平均处理时间: %.4f ms/帧\n' % avg_roi_time)
        f.write('每帧平均ROI数量: %.2f\n' % avg_rois)
        f.write('总处理帧数 (不含预热): %d\n' % total_processed_frames)
        f.write('总处理时间: %.4f ms\n' % (sum(head_processing_times) + sum(roi_processing_times)))
    
    # 复制配置文件到输出目录
    try:
        head_config_src = os.path.join('models', args.head_model, 'config.yml')
        head_config_dst = os.path.join(output_dir, f'{args.head_model}_config.yml')
        shutil.copy2(head_config_src, head_config_dst)
        
        roi_config_src = os.path.join('models', args.roi_model, 'config.yml')
        roi_config_dst = os.path.join(output_dir, f'{args.roi_model}_config.yml')
        shutil.copy2(roi_config_src, roi_config_dst)
        
        print(f"已复制配置文件到输出目录")
    except Exception as e:
        print(f"复制配置文件时出错: {e}")
    
    torch.cuda.empty_cache()
    
    print(f"分析后的视频已保存到: {output_video_path}")
    print(f"处理结果已保存到: {result_file}")

if __name__ == '__main__':
    main()
    