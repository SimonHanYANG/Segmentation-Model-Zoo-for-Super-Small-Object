'''
ENet TensorRT Video Inference Script
使用TensorRT引擎对视频进行实时分割推理


# 基本使用
python val_video_tensorrt.py --name ADSCNet_mermerHeadTail_0710 --engine_name model_fp32.engine --video_path /home/simon/UNeXt-pytorch/inputs/MermerTestVideo/mermerTest.avi

python val_video_tensorrt.py --name EDANet_sperm_NAHeadLatest250610_0610 --engine_name model_fp32.engine --video_path ~/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_121237814.mp4


'''
import argparse
import os
import time
import shutil
import numpy as np
from collections import deque

import cv2
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import yaml
from tqdm import tqdm
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize

# TensorRT 日志
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTInference:
    """TensorRT推理类"""
    def __init__(self, engine_path, num_classes):
        self.num_classes = num_classes
        
        # 加载引擎
        print(f"加载TensorRT引擎: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("加载TensorRT引擎失败")
        
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.input_binding_idx = 0
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_binding_idx = i
                break
        
        # 获取引擎实际输入形状
        self.engine_input_shape = tuple(self.engine.get_binding_shape(self.input_binding_idx))
        
        # 为了确保使用 batch_size=1，创建一个新的输入形状
        # 注意：保留原始输入形状的通道数、高度和宽度
        self.input_shape = (1, *self.engine_input_shape[1:])
        
        # 设置执行上下文的输入形状
        self.context.set_binding_shape(self.input_binding_idx, self.input_shape)
        
        # 分配GPU内存
        self.allocate_buffers()
        
        # 创建CUDA stream
        self.stream = cuda.Stream()
        
        print(f"TensorRT引擎加载成功")
        print(f"引擎输入形状: {self.engine_input_shape}")
        print(f"使用输入形状: {self.input_shape}")
        print(f"绑定数量: {self.engine.num_bindings}")
        
    def allocate_buffers(self):
        """分配GPU内存缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_shapes = []
        
        for binding_idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding_idx):
                # 使用自定义的 batch_size=1 输入形状
                shape = self.input_shape
            else:
                # 获取输出形状（确保与输入 batch_size 一致）
                shape = tuple(self.context.get_binding_shape(binding_idx))
                # 如果输出形状的第一维（batch_size）与输入不一致，调整它
                if shape[0] != self.input_shape[0]:
                    shape = (self.input_shape[0], *shape[1:])
            
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            # 计算所需内存大小
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            
            # 分配设备内存
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            
            # 分配主机内存
            host_mem = cuda.pagelocked_empty(shape, dtype)
            
            if self.engine.binding_is_input(binding_idx):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                self.output_shapes.append(shape)
    
    def preprocess_frame(self, frame, input_h, input_w):
        """预处理视频帧"""
        # 创建预处理管道 - 与val_video_pth.py保持一致
        transform = Compose([
            Resize(input_h, input_w),
            transforms.Normalize(),
        ])
        
        # 创建空掩码（仅用于转换）
        dummy_mask = np.zeros((frame.shape[0], frame.shape[1], self.num_classes), dtype=np.uint8)
        
        # 应用转换
        augmented = transform(image=frame, mask=dummy_mask)
        frame_transformed = augmented['image']
        
        # 转换为float32并归一化
        frame_normalized = frame_transformed.astype(np.float32) / 255.0
        
        # 转换为CHW格式
        frame_chw = np.transpose(frame_normalized, (2, 0, 1))
        
        # 添加batch维度
        frame_batch = np.expand_dims(frame_chw, axis=0)
        
        return frame_batch
    
    def infer(self, input_data):
        """执行推理"""
        # 检查输入形状
        if input_data.shape != self.input_shape:
            raise ValueError(f"输入形状不匹配: 得到 {input_data.shape}, 期望 {self.input_shape}")
            
        # 复制输入数据到主机内存
        np.copyto(self.inputs[0]['host'], input_data)
        
        # 传输输入数据到GPU
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 传输输出数据到主机
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output['host'],
                output['device'],
                self.stream
            )
        
        # 同步
        self.stream.synchronize()
        
        # 获取输出
        outputs = []
        for i, output in enumerate(self.outputs):
            output_data = output['host'].copy()  # 复制数据避免潜在问题
            outputs.append(output_data)
        
        return outputs[0] if len(outputs) == 1 else outputs

def parse_args():
    parser = argparse.ArgumentParser(description='使用TensorRT引擎对视频进行语义分割')
    
    parser.add_argument('--name', default=None, required=True,
                        help='模型名称 (包含model.engine和config.yml的文件夹)')
    parser.add_argument('--engine_name', default='model_fp32.engine',
                        help='引擎文件名称')
    parser.add_argument('--video_path', required=True,
                        help='输入视频文件路径')
    parser.add_argument('--output_fps', type=float, default=None,
                        help='输出视频FPS (默认: 与输入相同)')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行基准测试模式 (测量FPS)')
    parser.add_argument('--warmup_frames', type=int, default=10,
                        help='基准测试的预热帧数')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载配置文件
    config_path = os.path.join('models', args.name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print('-'*50)
    print('配置信息:')
    for key in config.keys():
        print(f'  {key}: {config[key]}')
    print('-'*50)
    
    # 构建引擎路径
    engine_path = os.path.join('models', args.name, args.engine_name)
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"引擎文件未找到: {engine_path}")
    
    # 设置输出目录
    output_dir = os.path.join('outputs', f"{args.name}_tensorrt")
    os.makedirs(output_dir, exist_ok=True)
    
    # 视频输入/输出路径
    video_filename = os.path.basename(args.video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_segmented_trt.mp4")
    
    # 打开视频
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {args.video_path}")
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fps = args.output_fps if args.output_fps else fps
    
    print(f"\n视频信息:")
    print(f"  尺寸: {frame_width}x{frame_height}")
    print(f"  FPS: {fps}")
    print(f"  总帧数: {total_frames}")
    print(f"  输出FPS: {output_fps}")
    
    # 创建TensorRT推理器
    trt_infer = TensorRTInference(engine_path, config['num_classes'])
    
    # 获取模型的输入大小
    input_h = config.get('input_h', trt_infer.input_shape[2])
    input_w = config.get('input_w', trt_infer.input_shape[3])
    print(f"使用模型输入尺寸: {input_h}x{input_w}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))
    
    # 定义叠加颜色（BGR格式）- 与val_video_pth.py保持一致
    overlay_colors = [
        (0, 0, 255),    # 类别0: 红色
        (0, 255, 0),    # 类别1: 绿色
        (255, 0, 0),    # 类别2: 蓝色
        (0, 255, 255)   # 类别3: 黄色
    ]
    opacity = 0.35  # 35%不透明度
    
    # 性能统计
    inference_times = []
    preprocessing_times = []
    postprocessing_times = []
    fps_window = deque(maxlen=30)  # 用于计算移动平均FPS
    
    # 处理视频
    print(f"\n开始处理视频...")
    frame_count = 0
    
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 总处理开始时间
            total_start = time.time()
            
            # 保存原始帧
            original_frame = frame.copy()
            
            # 预处理
            preprocess_start = time.time()
            input_tensor = trt_infer.preprocess_frame(frame, input_h, input_w)
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            # 推理
            inference_start = time.time()
            output = trt_infer.infer(input_tensor)
            inference_time = (time.time() - inference_start) * 1000
            
            # 后处理
            postprocess_start = time.time()
            
            # Sigmoid激活
            output_prob = 1 / (1 + np.exp(-output))  # sigmoid
            
            # 二值化
            output_binary = (output_prob >= 0.5).astype(np.float32)
            
            # 将预测结果叠加到原始帧上 - 与val_video_pth.py保持一致的处理方式
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
                
                # 叠加到原始帧上
                cv2.addWeighted(overlay, opacity, original_frame, 1, 0, original_frame)
            
            postprocess_time = (time.time() - postprocess_start) * 1000
            
            # 写入输出视频
            out.write(original_frame)
            
            # 记录性能数据（跳过预热帧）
            if frame_count >= args.warmup_frames:
                inference_times.append(inference_time)
                preprocessing_times.append(preprocess_time)
                postprocessing_times.append(postprocess_time)
                
                # 计算FPS
                total_time = time.time() - total_start
                current_fps = 1.0 / total_time
                fps_window.append(current_fps)
            
            frame_count += 1
            pbar.update(1)
            
            # 更新进度条信息
            if len(fps_window) > 0:
                avg_fps = sum(fps_window) / len(fps_window)
                pbar.set_postfix({
                    '推理时间': f'{inference_time:.2f}ms',
                    'FPS': f'{avg_fps:.1f}'
                })
    
    # 释放资源
    cap.release()
    out.release()
    
    # 计算统计数据
    if inference_times:
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        min_inference = np.min(inference_times)
        max_inference = np.max(inference_times)
        
        avg_preprocess = np.mean(preprocessing_times)
        avg_postprocess = np.mean(postprocessing_times)
        avg_total = avg_preprocess + avg_inference + avg_postprocess
        
        # 计算FPS
        inference_fps = 1000.0 / avg_inference
        avg_fps = 1000.0 / avg_total
        
        # 打印结果
        print('\n' + '='*60)
        print('TensorRT推理结果:')
        print('='*60)
        print(f'模型: {args.name}')
        print(f'引擎: {args.engine_name}')
        print(f'视频: {args.video_path}')
        print(f'输出: {output_video_path}')
        print(f'处理帧数: {len(inference_times)} (不包括 {args.warmup_frames} 预热帧)')
        print('\n性能统计:')
        print(f'  预处理:    {avg_preprocess:.2f} ms/帧')
        print(f'  推理:      {avg_inference:.2f} ± {std_inference:.2f} ms/帧')
        print(f'            最小: {min_inference:.2f} ms, 最大: {max_inference:.2f} ms')
        print(f'  后处理:    {avg_postprocess:.2f} ms/帧')
        print(f'  总时间:    {avg_total:.2f} ms/帧')
        print(f'  Inference FPS:       {inference_fps:.1f}')
        print(f'  FPS:       {avg_fps:.1f}')
        
        # 获取PyTorch模型的处理时间（如果可用）
        pytorch_result_file = os.path.join('outputs', args.name, f"{video_name}_processing_results.txt")
        pytorch_time = None
        if os.path.exists(pytorch_result_file):
            try:
                with open(pytorch_result_file, 'r') as f:
                    for line in f:
                        if "Average Processing Time:" in line:
                            parts = line.split(":")
                            if len(parts) > 1:
                                pytorch_time = float(parts[1].split()[0])
                                break
            except Exception as e:
                print(f"无法读取PyTorch结果文件: {e}")
        
        # 对比PyTorch模型性能
        print('\n加速分析:')
        if pytorch_time:
            speedup = pytorch_time / avg_inference
            print(f'  相比PyTorch模型加速: {speedup:.1f}x')
        else:
            # 如果没有PyTorch结果，使用预估值
            pytorch_time = 35.0  # 预估值，需根据实际情况调整
            speedup = pytorch_time / avg_inference
            print(f'  相比PyTorch模型预估加速: {speedup:.1f}x (基于预估PyTorch时间 {pytorch_time:.1f}ms)')
        print('='*60)
        
        # 保存结果到文件
        result_file = os.path.join(output_dir, f"{video_name}_trt_results.txt")
        with open(result_file, 'w') as f:
            f.write('TensorRT推理结果\n')
            f.write('='*60 + '\n')
            f.write(f'模型: {args.name}\n')
            f.write(f'引擎: {args.engine_name}\n')
            f.write(f'视频: {args.video_path}\n')
            f.write(f'输出: {output_video_path}\n')
            f.write(f'输入尺寸: {input_h}x{input_w}\n')
            f.write(f'批次大小: 1\n')
            f.write(f'处理帧数: {len(inference_times)}\n')
            f.write(f'预热帧数: {args.warmup_frames}\n')
            f.write('\n性能统计:\n')
            f.write(f'  预处理:    {avg_preprocess:.2f} ms/帧\n')
            f.write(f'  推理:      {avg_inference:.2f} ± {std_inference:.2f} ms/帧\n')
            f.write(f'            最小: {min_inference:.2f} ms, 最大: {max_inference:.2f} ms\n')
            f.write(f'  后处理:    {avg_postprocess:.2f} ms/帧\n')
            f.write(f'  总时间:    {avg_total:.2f} ms/帧\n')
            f.write(f'  FPS:       {avg_fps:.1f}\n')
            if pytorch_time:
                f.write(f'\n相比PyTorch模型加速: {speedup:.1f}x (PyTorch: {pytorch_time:.2f}ms)\n')
            else:
                f.write(f'\n相比PyTorch模型预估加速: {speedup:.1f}x (预估PyTorch: {pytorch_time:.2f}ms)\n')
            
            # 添加分位数统计
            percentiles = [50, 90, 95, 99]
            f.write('\n推理时间分位数:\n')
            for p in percentiles:
                val = np.percentile(inference_times, p)
                f.write(f'  {p}分位数: {val:.2f} ms\n')
        
        print(f"\n结果已保存到: {result_file}")
        
        # 复制配置文件
        try:
            shutil.copy2(config_path, os.path.join(output_dir, 'config.yml'))
            print(f"配置文件已复制到: {output_dir}")
        except Exception as e:
            print(f"复制配置文件时出错: {e}")
    
    print(f"\n分割视频已保存到: {output_video_path}")
    
    # 基准测试模式
    if args.benchmark:
        print("\n" + "="*60)
        print("运行基准测试模式...")
        print("="*60)
        
        # 预热
        print(f"使用 {args.warmup_frames} 帧进行预热...")
        dummy_input = np.zeros(trt_infer.input_shape, dtype=np.float32)
        for _ in range(args.warmup_frames):
            _ = trt_infer.infer(dummy_input)
        
        # 基准测试
        num_iterations = 100
        print(f"运行 {num_iterations} 次迭代...")
        
        benchmark_times = []
        for _ in tqdm(range(num_iterations)):
            start = time.time()
            _ = trt_infer.infer(dummy_input)
            benchmark_times.append((time.time() - start) * 1000)
        
        # 打印基准测试结果
        avg_time = np.mean(benchmark_times)
        std_time = np.std(benchmark_times)
        min_time = np.min(benchmark_times)
        max_time = np.max(benchmark_times)
        
        print(f"\n基准测试结果:")
        print(f"  平均: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  最小: {min_time:.2f} ms")
        print(f"  最大: {max_time:.2f} ms")
        print(f"  吞吐量: {1000/avg_time:.1f} FPS")
        print("="*60)

if __name__ == '__main__':
    main()