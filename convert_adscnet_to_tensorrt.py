#!/usr/bin/env python3
"""
Convert ADSCNet PyTorch model to ONNX and TensorRT
Author: Simon
Date: 2025/07/24

python convert_adscnet_to_tensorrt.py --model_dir models/ADSCNet_mermerHeadTail_0710
python convert_adscnet_to_tensorrt.py --model_dir models/ADSCNet_mermerHeadTail_0710 --precision fp16
"""

import os
import argparse
import yaml
import torch
import numpy as np
import tensorrt as trt
from model_zoo.adscnet import ADSCNet
from modules import conv1x1, ConvBNAct, DWConvBNAct, DeConvBNAct  # Needed for model loading

# Set up TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_model(model_path, config_path):
    """Load PyTorch model using configuration"""
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model instance
    model = ADSCNet(
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision']
    )
    
    # Load trained weights
    print(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, config

def convert_to_onnx(model, config, onnx_path):
    """Convert PyTorch model to ONNX format"""
    print(f"Converting model to ONNX format: {onnx_path}")
    
    # Create dummy input
    batch_size = 1
    channels = config['input_channels']
    height = config['input_h']
    width = config['input_w']
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # Export to ONNX
    torch.onnx.export(
        model,                     # PyTorch model
        dummy_input,               # Input tensor
        onnx_path,                 # Output file
        export_params=True,        # Store trained weights
        opset_version=13,          # ONNX opset version
        do_constant_folding=True,  # Optimize constants
        input_names=['input'],     # Input tensor name
        output_names=['output'],   # Output tensor name
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"Model exported to ONNX successfully: {onnx_path}")
    return onnx_path

def build_engine(onnx_path, engine_path, precision="fp32", workspace_size=1<<30, config=None):
    """Convert ONNX model to TensorRT engine"""
    print(f"Building TensorRT engine: {engine_path}")
    print(f"Using precision: {precision}")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX file
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")
    
    # Configure builder
    config_builder = builder.create_builder_config()
    
    # 使用新的API设置工作空间大小
    config_builder.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    # 设置精度模式
    if precision.lower() == "fp16" and builder.platform_has_fast_fp16:
        config_builder.set_flag(trt.BuilderFlag.FP16)
        print("Enabled FP16 precision")
    elif precision.lower() == "int8" and builder.platform_has_fast_int8:
        config_builder.set_flag(trt.BuilderFlag.INT8)
        # Would need calibration data for INT8
        print("Enabled INT8 precision (note: calibration is required for optimal accuracy)")
    
    # 创建优化配置文件来处理动态输入
    profile = builder.create_optimization_profile()
    
    # 获取输入尺寸
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    channels = input_shape[1]
    
    # 从配置文件获取输入尺寸，如果配置可用
    height = width = 512  # 默认值
    if config and 'input_h' in config and 'input_w' in config:
        height = config['input_h']
        width = config['input_w']
    
    # 设置最小、最佳和最大输入形状
    min_batch = 1
    opt_batch = 1
    max_batch = 8  # 可以根据需要调整最大批次大小
    
    profile.set_shape(
        input_name, 
        (min_batch, channels, height, width),      # 最小形状
        (opt_batch, channels, height, width),      # 最佳形状
        (max_batch, channels, height, width)       # 最大形状
    )
    
    config_builder.add_optimization_profile(profile)
    
    # 构建和保存引擎
    serialized_engine = builder.build_serialized_network(network, config_builder)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine built and saved to: {engine_path}")
    return engine_path

def verify_engine(engine_path):
    """Load and verify the TensorRT engine"""
    print(f"Verifying TensorRT engine: {engine_path}")
    
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError("Failed to load TensorRT engine")
    
    # 对于动态输入形状的引擎，绑定形状可能会显示为-1
    print(f"Engine successfully verified.")
    print(f"Number of bindings: {engine.num_bindings}")
    
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        print(f"Binding {i}: {name} - Shape: {shape}")
        
        # 如果有动态维度，它们可能显示为-1
        if -1 in shape:
            print(f"  Note: Binding {name} has dynamic dimensions")
            profile_count = engine.num_optimization_profiles
            print(f"  Engine has {profile_count} optimization profile(s)")
            
            # 尝试获取第一个优化配置文件的信息
            if profile_count > 0:
                min_shape = engine.get_profile_shape(0, i)[0]
                opt_shape = engine.get_profile_shape(0, i)[1]
                max_shape = engine.get_profile_shape(0, i)[2]
                print(f"  Profile 0 shapes - Min: {min_shape}, Opt: {opt_shape}, Max: {max_shape}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert ADSCNet PyTorch model to ONNX and TensorRT')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model.pth and config.yml')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for ONNX and TensorRT models')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'], 
                        help='Precision for TensorRT engine (fp32, fp16, int8)')
    args = parser.parse_args()
    
    # Set up paths
    model_path = os.path.join(args.model_dir, 'model.pth')
    config_path = os.path.join(args.model_dir, 'config.yml')
    
    if args.output_dir is None:
        args.output_dir = args.model_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    onnx_path = os.path.join(args.output_dir, 'model.onnx')
    engine_path = os.path.join(args.output_dir, f'model_{args.precision}.engine')
    
    # Check if model and config exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load PyTorch model
    model, config = load_model(model_path, config_path)
    
    # Convert to ONNX
    convert_to_onnx(model, config, onnx_path)
    
    # Convert to TensorRT
    build_engine(onnx_path, engine_path, args.precision, config=config)
    
    # Verify engine
    verify_engine(engine_path)
    
    print("Conversion completed successfully!")
    print(f"- PyTorch model: {model_path}")
    print(f"- ONNX model: {onnx_path}")
    print(f"- TensorRT engine ({args.precision}): {engine_path}")

if __name__ == '__main__':
    main()