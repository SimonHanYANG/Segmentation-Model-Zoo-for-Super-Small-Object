"""
TensorRT Folder Inference Script
使用TensorRT引擎对文件夹中的图像进行批量分割预测

# 基本使用
python val-folder-tensorrt.py \
    --name EDANet_sperm_ROINAHead_250707 \
    --val_folder inputs/20250707-UNeXt-ROI-NA-dataset/images

# 使用FP16引擎
python val-folder-tensorrt.py \
    --name EDANet_sperm_ROINAHead_250707 \
    --engine_name model_fp16.engine \
    --val_folder inputs/20250707-UNeXt-ROI-NA-dataset/images

# 运行基准测试模式
python val-folder-tensorrt.py \
    --name EDANet_sperm_ROINAHead_250707 \
    --val_folder inputs/20250707-UNeXt-ROI-NA-dataset/images \
    --benchmark
"""
import argparse
import os
import time
import shutil
import numpy as np
from glob import glob

import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import yaml
from tqdm import tqdm

# TensorRT 日志
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTInference:
    """TensorRT推理类"""
    def __init__(self, engine_path, num_classes):
        self.num_classes = num_classes
        
        # 加载引擎
        print(f"Loading TensorRT engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
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
        self.input_shape = (1, *self.engine_input_shape[1:])
        
        # 设置执行上下文的输入形状
        self.context.set_binding_shape(self.input_binding_idx, self.input_shape)
        
        # 分配GPU内存
        self.allocate_buffers()
        
        # 创建CUDA stream
        self.stream = cuda.Stream()
        
        print(f"TensorRT engine loaded successfully")
        print(f"Engine input shape: {self.engine_input_shape}")
        print(f"Using input shape: {self.input_shape}")
        print(f"Number of bindings: {self.engine.num_bindings}")
        
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
    
    def preprocess_image(self, image, input_h, input_w):
        """预处理图像"""
        # 调整大小
        image_resized = cv2.resize(image, (input_w, input_h))
        
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 转换为CHW格式
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        
        # 添加batch维度
        image_batch = np.expand_dims(image_chw, axis=0)
        
        return image_batch.astype(np.float32)
    
    def infer(self, input_data):
        """执行推理"""
        # 检查输入形状
        if input_data.shape != self.input_shape:
            raise ValueError(f"Input shape mismatch: got {input_data.shape}, expected {self.input_shape}")
            
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

def get_image_files(folder_path, specific_ext=None):
    """获取文件夹中的所有图像文件，可以指定特定扩展名或自动检测"""
    if specific_ext:
        # 使用指定的扩展名
        img_files = glob(os.path.join(folder_path, f'*{specific_ext}'))
        return img_files, specific_ext
    
    # 常见图像扩展名
    common_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 先尝试常见扩展名
    for ext in common_exts:
        img_files = glob(os.path.join(folder_path, f'*{ext}'))
        if img_files:
            return img_files, ext
    
    # 如果没有找到常见扩展名的图像，尝试查找任何文件并推测
    all_files = glob(os.path.join(folder_path, '*'))
    
    # 过滤可能的图像文件（基于扩展名）
    img_files = [f for f in all_files if os.path.splitext(f)[1].lower() in common_exts]

    if not img_files:
        raise ValueError(f"在 {folder_path} 中没有找到图像文件")
    
    # 使用第一个图像文件的扩展名
    detected_ext = os.path.splitext(img_files[0])[1]
    
    # 重新查找具有相同扩展名的所有文件
    img_files = glob(os.path.join(folder_path, f'*{detected_ext}'))
    
    return img_files, detected_ext

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default=None, required=True,
                        help='model name (folder containing model.engine and config.yml)')
    parser.add_argument('--engine_name', default='model_fp32.engine',
                        help='name of the engine file')
    parser.add_argument('--val_folder', required=True,
                        help='path to folder containing images')
    parser.add_argument('--img_ext', default=None,
                        help='image extension to use (e.g., .jpg, .png); if not specified, auto-detect')
    parser.add_argument('--output_size', default='1600,1200',
                        help='output image size in format "width,height"')
    parser.add_argument('--benchmark', action='store_true',
                        help='run benchmark mode (measure FPS)')
    parser.add_argument('--warmup_iterations', type=int, default=10,
                        help='number of warmup iterations for benchmarking')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载配置文件
    config_path = os.path.join('models', args.name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print('-'*50)
    print('Configuration:')
    for key in config.keys():
        print(f'  {key}: {config[key]}')
    print('-'*50)
    
    # 构建引擎路径
    engine_path = os.path.join('models', args.name, args.engine_name)
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")
    
    # 解析输出尺寸
    try:
        OUTPUT_WIDTH, OUTPUT_HEIGHT = map(int, args.output_size.split(','))
    except:
        # 默认尺寸
        OUTPUT_WIDTH, OUTPUT_HEIGHT = 1600, 1200
        print(f"使用默认输出尺寸: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    
    # 设置输出目录
    output_base_dir = os.path.join('outputs', f"{args.name}_tensorrt", os.path.basename(args.val_folder.rstrip('/')))
    
    # 为每个类别创建输出目录
    for c in range(config['num_classes']):
        os.makedirs(os.path.join(output_base_dir, str(c)), exist_ok=True)
    
    # 获取图像文件列表
    img_files, detected_ext = get_image_files(args.val_folder, args.img_ext)
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_files]
    
    print(f"\n找到 {len(img_files)} 张图像, 扩展名: {detected_ext}")
    
    # 创建TensorRT推理器
    trt_infer = TensorRTInference(engine_path, config['num_classes'])
    
    # 获取模型的实际输入大小
    input_h = trt_infer.input_shape[2]
    input_w = trt_infer.input_shape[3]
    print(f"使用模型输入尺寸: {input_h}x{input_w}")
    
    # 处理性能统计
    inference_times = []
    preprocessing_times = []
    postprocessing_times = []
    
    # 进行预热
    print(f"\n预热 {args.warmup_iterations} 次...")
    dummy_input = np.zeros(trt_infer.input_shape, dtype=np.float32)
    for _ in range(args.warmup_iterations):
        _ = trt_infer.infer(dummy_input)
    
    # 处理所有图像
    print(f"\n开始处理 {len(img_files)} 张图像...")
    
    with tqdm(total=len(img_files)) as pbar:
        for img_path, img_id in zip(img_files, img_ids):
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}, 跳过")
                continue
            
            # 记录原始图像尺寸
            original_h, original_w = img.shape[:2]
            
            # 预处理
            preprocess_start = time.time()
            input_tensor = trt_infer.preprocess_image(img, input_h, input_w)
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
            
            # 保存结果
            for c in range(config['num_classes']):
                # 调整预测结果大小到指定的输出尺寸
                mask_resized = cv2.resize(
                    output_binary[0, c],
                    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # 转换为8位灰度图像
                mask_8bit = (mask_resized * 255).astype(np.uint8)
                
                # 保存掩码图像
                save_path = os.path.join(output_base_dir, str(c), f"{img_id}.png")
                cv2.imwrite(save_path, mask_8bit)
                
                # 可选：保存概率图
                if config.get('save_probability_maps', False):
                    prob_resized = cv2.resize(
                        output_prob[0, c],
                        (OUTPUT_WIDTH, OUTPUT_HEIGHT)
                    )
                    prob_8bit = (prob_resized * 255).astype(np.uint8)
                    prob_save_path = os.path.join(output_base_dir, str(c), f"{img_id}_prob.png")
                    cv2.imwrite(prob_save_path, prob_8bit)
            
            postprocess_time = (time.time() - postprocess_start) * 1000
            
            # 记录性能数据
            inference_times.append(inference_time)
            preprocessing_times.append(preprocess_time)
            postprocessing_times.append(postprocess_time)
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'Inference': f'{inference_time:.2f}ms',
            })
    
    # 计算并保存性能统计
    if inference_times:
        avg_inference = np.mean(inference_times)
        std_inference = np.std(inference_times)
        min_inference = np.min(inference_times)
        max_inference = np.max(inference_times)
        
        avg_preprocess = np.mean(preprocessing_times)
        avg_postprocess = np.mean(postprocessing_times)
        avg_total = avg_preprocess + avg_inference + avg_postprocess
        
        # 计算FPS
        avg_fps = 1000.0 / avg_total
        
        # 打印结果
        print('\n' + '='*60)
        print('TensorRT Inference Results:')
        print('='*60)
        print(f'Model: {args.name}')
        print(f'Engine: {args.engine_name}')
        print(f'Images Folder: {args.val_folder}')
        print(f'Output Folder: {output_base_dir}')
        print(f'Images processed: {len(inference_times)}')
        print('\nPerformance Statistics:')
        print(f'  Preprocessing:  {avg_preprocess:.2f} ms/image')
        print(f'  Inference:      {avg_inference:.2f} ± {std_inference:.2f} ms/image')
        print(f'                  Min: {min_inference:.2f} ms, Max: {max_inference:.2f} ms')
        print(f'  Postprocessing: {avg_postprocess:.2f} ms/image')
        print(f'  Total:          {avg_total:.2f} ms/image')
        print(f'  Throughput:     {avg_fps:.1f} images/sec')
        print('='*60)
        
        # 保存结果到文件
        result_file = os.path.join(output_base_dir, "tensorrt_results.txt")
        with open(result_file, 'w') as f:
            f.write('TensorRT Inference Results\n')
            f.write('='*60 + '\n')
            f.write(f'Model: {args.name}\n')
            f.write(f'Engine: {args.engine_name}\n')
            f.write(f'Images Folder: {args.val_folder}\n')
            f.write(f'Output Folder: {output_base_dir}\n')
            f.write(f'Output Size: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}\n')
            f.write(f'Input Size: {input_h}x{input_w}\n')
            f.write(f'Images processed: {len(inference_times)}\n')
            f.write(f'Warmup iterations: {args.warmup_iterations}\n')
            f.write('\nPerformance Statistics:\n')
            f.write(f'  Preprocessing:  {avg_preprocess:.2f} ms/image\n')
            f.write(f'  Inference:      {avg_inference:.2f} ± {std_inference:.2f} ms/image\n')
            f.write(f'                  Min: {min_inference:.2f} ms, Max: {max_inference:.2f} ms\n')
            f.write(f'  Postprocessing: {avg_postprocess:.2f} ms/image\n')
            f.write(f'  Total:          {avg_total:.2f} ms/image\n')
            f.write(f'  Throughput:     {avg_fps:.1f} images/sec\n')
            
            # 添加分位数统计
            percentiles = [50, 90, 95, 99]
            f.write('\nInference Time Percentiles:\n')
            for p in percentiles:
                val = np.percentile(inference_times, p)
                f.write(f'  {p}th percentile: {val:.2f} ms\n')
            
            # 保存每张图片的处理时间
            f.write('\nDetailed Processing Times (ms):\n')
            f.write('Image ID,Preprocessing,Inference,Postprocessing,Total\n')
            for i, (img_id, pre, inf, post) in enumerate(zip(img_ids, preprocessing_times, inference_times, postprocessing_times)):
                total = pre + inf + post
                f.write(f'{img_id},{pre:.2f},{inf:.2f},{post:.2f},{total:.2f}\n')
        
        # 保存CSV格式的详细性能数据
        csv_file = os.path.join(output_base_dir, "processing_times.csv")
        with open(csv_file, 'w') as f:
            f.write('image_id,preprocessing_ms,inference_ms,postprocessing_ms,total_ms\n')
            for i, (img_id, pre, inf, post) in enumerate(zip(img_ids, preprocessing_times, inference_times, postprocessing_times)):
                total = pre + inf + post
                f.write(f'{img_id},{pre:.2f},{inf:.2f},{post:.2f},{total:.2f}\n')
        
        # 复制配置文件
        try:
            shutil.copy2(config_path, os.path.join(output_base_dir, 'config.yml'))
            print(f"配置文件已复制到: {output_base_dir}")
        except Exception as e:
            print(f"复制配置文件时出错: {e}")
        
        print(f"\n结果已保存到: {result_file}")
        print(f"详细处理时间已保存到: {csv_file}")
    
    # Benchmark模式
    if args.benchmark:
        print("\n" + "="*60)
        print("运行基准测试模式...")
        print("="*60)
        
        # 预热
        print(f"预热 {args.warmup_iterations} 次...")
        dummy_input = np.zeros(trt_infer.input_shape, dtype=np.float32)
        for _ in range(args.warmup_iterations):
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
        print(f"  平均耗时: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  最小耗时: {min_time:.2f} ms")
        print(f"  最大耗时: {max_time:.2f} ms")
        print(f"  吞吐量: {1000/avg_time:.1f} 图像/秒")
        
        # 保存基准测试结果
        benchmark_file = os.path.join(output_base_dir, "benchmark_results.txt")
        with open(benchmark_file, 'w') as f:
            f.write('TensorRT Benchmark Results\n')
            f.write('='*60 + '\n')
            f.write(f'Model: {args.name}\n')
            f.write(f'Engine: {args.engine_name}\n')
            f.write(f'Iterations: {num_iterations}\n')
            f.write(f'Warmup Iterations: {args.warmup_iterations}\n')
            f.write(f'Input Size: {input_h}x{input_w}\n')
            f.write('\nResults:\n')
            f.write(f'  Average: {avg_time:.2f} ± {std_time:.2f} ms\n')
            f.write(f'  Min: {min_time:.2f} ms\n')
            f.write(f'  Max: {max_time:.2f} ms\n')
            f.write(f'  Throughput: {1000/avg_time:.1f} images/sec\n')
            
            # 添加分位数统计
            percentiles = [50, 90, 95, 99]
            f.write('\nPercentiles:\n')
            for p in percentiles:
                val = np.percentile(benchmark_times, p)
                f.write(f'  {p}th percentile: {val:.2f} ms\n')
            
            # 保存所有迭代的时间
            f.write('\nDetailed Times (ms):\n')
            for i, t in enumerate(benchmark_times):
                f.write(f'Iteration {i+1}: {t:.2f}\n')
        
        print(f"\n基准测试结果已保存到: {benchmark_file}")
        print("="*60)

if __name__ == '__main__':
    main()