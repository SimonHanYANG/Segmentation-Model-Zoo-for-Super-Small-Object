"""
ENet PyTorch to TensorRT Conversion Script
通过修改 ENet 模型替换 MaxUnpool2d 来支持 TensorRT 转换

# 基本使用
python convert_enet_to_tensorrt.py --model_dir models/ENet_mermerHeadTail_0710

# 使用FP16精度
python convert_enet_to_tensorrt.py --model_dir models/ENet_mermerHeadTail_0710 --fp16

# 启用动态批处理
python convert_enet_to_tensorrt.py \
    --model_dir models/ENet_mermerHeadTail_0710 \
    --dynamic_batch \
    --min_batch 1 \
    --max_batch 8 \
    --fp16

# 保留ONNX文件并启用详细日志
python convert_enet_to_tensorrt.py \
    --model_dir models/ENet_mermerHeadTail_0710 \
    --keep_onnx \
    --verbose
"""

import os
import argparse
import torch
import tensorrt as trt
import numpy as np
from collections import OrderedDict
import yaml
import torch.nn as nn
import torch.nn.functional as F

# 导入必要的模块
import sys
sys.path.append('.')
from modules import conv1x1, ConvBNAct, Activation

# TensorRT日志级别
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ============== 修改版的 ENet 组件 ==============

class ModifiedBottleneck(nn.Module):
    """修改版的 Bottleneck，使用双线性插值代替 MaxUnpool2d"""
    def __init__(self, in_channels, out_channels, conv_type, act_type='prelu', 
                    upsample_type='regular', dilation=1, drop_p=0.1, shrink_ratio=0.25):
        super().__init__()
        self.conv_type = conv_type
        hid_channels = int(in_channels * shrink_ratio)
        
        if conv_type == 'regular':
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 1),
                ConvBNAct(hid_channels, hid_channels),
            )
        elif conv_type == 'downsampling':
            # 保持下采样不变，但不使用 return_indices
            self.left_pool = nn.MaxPool2d(2, 2, return_indices=False)
            self.left_conv = ConvBNAct(in_channels, out_channels, 1)         
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 3, 2),
                ConvBNAct(hid_channels, hid_channels),
            )
        elif conv_type == 'upsampling':
            self.left_conv = ConvBNAct(in_channels, out_channels, 1)
            # 使用双线性插值代替 MaxUnpool2d
            self.left_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 1),
                ModifiedUpsample(hid_channels, hid_channels, scale_factor=2,  
                         kernel_size=3, upsample_type=upsample_type),
            )
        elif conv_type == 'dilate':
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 1),
                ConvBNAct(hid_channels, hid_channels, dilation=dilation),
            )
        elif conv_type == 'asymmetric':
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 1),
                ConvBNAct(hid_channels, hid_channels, (5,1)),
                ConvBNAct(hid_channels, hid_channels, (1,5)),
            )
        else:
            raise ValueError(f'[!] Unsupport convolution type: {conv_type}')

        self.right_last_conv = nn.Sequential(
            conv1x1(hid_channels, out_channels),
            nn.Dropout(drop_p)
        )
        self.act = Activation(act_type)

    def forward(self, x, indices=None):
        x_right = self.right_last_conv(self.right_init_conv(x))
        
        if self.conv_type == 'downsampling':
            x_left = self.left_pool(x)
            x_left = self.left_conv(x_left)
            x = self.act(x_left + x_right)
            return x, None  # 返回 None 代替 indices
        elif self.conv_type == 'upsampling':
            x_left = self.left_conv(x)
            x_left = self.left_upsample(x_left)
            x = self.act(x_left + x_right)
        else:
            x = self.act(x + x_right)
        
        return x

class ModifiedUpsample(nn.Module):
    """修改版的 Upsample，确保兼容性"""
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, 
                    upsample_type=None, act_type='relu'):
        super().__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor - 1
            padding = (kernel_size - 1) // 2
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                stride=scale_factor, padding=padding, 
                                                output_padding=1, bias=False)
        else:
            self.up_conv = nn.Sequential(
                ConvBNAct(in_channels, out_channels, 1, act_type=act_type),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            )

    def forward(self, x):
        return self.up_conv(x)

class ModifiedBottleNeck1(nn.Module):
    """修改版的 BottleNeck1"""
    def __init__(self, in_channels, out_channels, act_type='prelu', drop_p=0.01):
        super().__init__()
        self.conv_pool = ModifiedBottleneck(in_channels, out_channels, 'downsampling', act_type, drop_p=drop_p)
        self.conv_regular = nn.Sequential(
            ModifiedBottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            ModifiedBottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            ModifiedBottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            ModifiedBottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
        )

    def forward(self, x):
        x, indices = self.conv_pool(x)
        x = self.conv_regular(x)
        return x, indices

class ModifiedBottleNeck23(nn.Module):
    """修改版的 BottleNeck23"""
    def __init__(self, in_channels, out_channels, act_type='prelu', downsample=True):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.conv_pool = ModifiedBottleneck(in_channels, out_channels, 'downsampling', act_type=act_type)

        self.conv_regular = nn.Sequential(
            ModifiedBottleneck(out_channels, out_channels, 'regular', act_type),
            ModifiedBottleneck(out_channels, out_channels, 'dilate', act_type, dilation=2),
            ModifiedBottleneck(out_channels, out_channels, 'asymmetric', act_type),
            ModifiedBottleneck(out_channels, out_channels, 'dilate', act_type, dilation=4),
            ModifiedBottleneck(out_channels, out_channels, 'regular', act_type),
            ModifiedBottleneck(out_channels, out_channels, 'dilate', act_type, dilation=8),
            ModifiedBottleneck(out_channels, out_channels, 'asymmetric', act_type),
            ModifiedBottleneck(out_channels, out_channels, 'dilate', act_type, dilation=16),
        )

    def forward(self, x):
        if self.downsample:
            x, indices = self.conv_pool(x)
        x = self.conv_regular(x)

        if self.downsample:
            return x, indices
        return x

class ModifiedBottleNeck45(nn.Module):
    """修改版的 BottleNeck45，使用修改后的 Bottleneck"""
    def __init__(self, in_channels, out_channels, act_type='prelu', upsample_type=None, 
                    extra_conv=False):
        super().__init__()
        self.extra_conv = extra_conv
        self.conv_unpool = ModifiedBottleneck(in_channels, out_channels, 'upsampling', act_type, upsample_type)
        self.conv_regular = ModifiedBottleneck(out_channels, out_channels, 'regular', act_type)
        
        if extra_conv:
            self.conv_extra = ModifiedBottleneck(out_channels, out_channels, 'regular', act_type)

    def forward(self, x, indices=None):
        x = self.conv_unpool(x, indices)
        x = self.conv_regular(x)
        
        if self.extra_conv:
            x = self.conv_extra(x)
        
        return x

class ModifiedInitialBlock(nn.Module):
    """修改版的 InitialBlock"""
    def __init__(self, in_channels, out_channels, act_type, kernel_size=3, **kwargs):
        super().__init__()
        assert out_channels > in_channels, 'out_channels should be larger than in_channels.\n'
        self.conv = ConvBNAct(in_channels, out_channels - in_channels, kernel_size, 2, act_type=act_type, **kwargs)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return x

class ENetForTensorRT(nn.Module):
    """修改版的 ENet，适用于 TensorRT 转换"""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 act_type='prelu', upsample_type='deconvolution', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 初始块
        self.initial = ModifiedInitialBlock(input_channels, 16, act_type)
        
        # 编码器部分
        self.bottleneck1 = ModifiedBottleNeck1(16, 64, act_type)
        self.bottleneck2 = ModifiedBottleNeck23(64, 128, act_type, True)
        self.bottleneck3 = ModifiedBottleNeck23(128, 128, act_type, False)
        
        # 解码器部分 - 使用修改版的 BottleNeck45
        self.bottleneck4 = ModifiedBottleNeck45(128, 64, act_type, upsample_type, True)
        self.bottleneck5 = ModifiedBottleNeck45(64, 16, act_type, upsample_type, False)
        self.fullconv = ModifiedUpsample(16, num_classes, scale_factor=2, act_type=act_type)
        
        # 深度监督分支
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(64, 32, 3, 1, act_type=act_type),
                conv1x1(32, num_classes)
            )
            self.aux_head2 = nn.Sequential(
                ConvBNAct(128, 64, 3, 1, act_type=act_type),
                conv1x1(64, num_classes)
            )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 初始块
        x = self.initial(x)
        
        # 编码器阶段1
        x, _ = self.bottleneck1(x)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        # 编码器阶段2
        x, _ = self.bottleneck2(x)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 编码器阶段3
        x = self.bottleneck3(x)
        
        # 解码器阶段
        x = self.bottleneck4(x, None)
        x = self.bottleneck5(x, None)
        x = self.fullconv(x)
        
        # 确保输出尺寸与输入一致
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x

# ============== 转换脚本主体 ==============

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ENet PyTorch model to TensorRT engine')
    
    # 模型文件夹路径（包含model.pth和config.yml）
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the model directory containing model.pth and config.yml')
    parser.add_argument('--model_name', type=str, default='model.pth',
                        help='Name of the model file (default: model.pth)')
    parser.add_argument('--engine_name', type=str, default='model.engine',
                        help='Name of the output engine file (default: model.engine)')
    
    # TensorRT配置
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: use training batch_size)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--int8', action='store_true',
                        help='Use INT8 precision (requires calibration)')
    parser.add_argument('--workspace_size', type=int, default=1,
                        help='Workspace size in GB')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    # ONNX中间文件
    parser.add_argument('--keep_onnx', action='store_true',
                        help='Keep the intermediate ONNX file')
    
    # 动态批处理配置
    parser.add_argument('--dynamic_batch', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--min_batch', type=int, default=1,
                        help='Minimum batch size for dynamic batching')
    parser.add_argument('--max_batch', type=int, default=8,
                        help='Maximum batch size for dynamic batching')
    
    return parser.parse_args()

def load_config(config_path):
    """加载训练时的配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Loaded training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    return config

def load_enet_model_with_conversion(model_path, config):
    """加载原始 ENet 模型并转换权重到修改版模型"""
    print(f"\nLoading and converting ENet model from {model_path}")
    
    # 导入原始 ENet
    from model_zoo.enet import ENet
    
    # 从配置中获取参数
    act_type = config.get('act_type', 'prelu')
    upsample_type = config.get('upsample_type', 'deconvolution')
    
    # 创建原始模型以加载权重
    original_model = ENet(
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        act_type=act_type,
        upsample_type=upsample_type
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 处理 DataParallel 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    original_model.load_state_dict(new_state_dict)
    
    # 创建修改版模型
    modified_model = ENetForTensorRT(
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        act_type=act_type,
        upsample_type=upsample_type
    )
    
    # 复制权重
    print("Transferring weights from original model to modified model...")
    modified_state_dict = modified_model.state_dict()
    original_state_dict = original_model.state_dict()
    
    # 智能复制权重
    transferred_keys = []
    skipped_keys = []
    
    for key in modified_state_dict.keys():
        if key in original_state_dict:
            if modified_state_dict[key].shape == original_state_dict[key].shape:
                modified_state_dict[key] = original_state_dict[key]
                transferred_keys.append(key)
            else:
                skipped_keys.append(f"{key}: shape mismatch")
        else:
            # 尝试找到对应的键（可能有命名差异）
            found = False
            for orig_key in original_state_dict.keys():
                if orig_key.replace('bottleneck', 'bottleneck').replace('conv_unpool', 'conv_unpool') == key:
                    if modified_state_dict[key].shape == original_state_dict[orig_key].shape:
                        modified_state_dict[key] = original_state_dict[orig_key]
                        transferred_keys.append(f"{key} <- {orig_key}")
                        found = True
                        break
            if not found:
                skipped_keys.append(f"{key}: not found in original model")
    
    modified_model.load_state_dict(modified_state_dict)
    modified_model.eval()
    
    print(f"\nWeight transfer summary:")
    print(f"  Total parameters in modified model: {len(modified_state_dict)}")
    print(f"  Successfully transferred: {len(transferred_keys)}")
    print(f"  Skipped/Default initialized: {len(skipped_keys)}")
    
    if skipped_keys and len(skipped_keys) < 20:  # 只显示少量跳过的键
        print("\n  Skipped keys:")
        for key in skipped_keys[:20]:
            print(f"    - {key}")
    
    print(f"\nModel converted successfully!")
    print(f"  Architecture: {config['arch']} (Modified for TensorRT)")
    print(f"  Input channels: {config['input_channels']}")
    print(f"  Number of classes: {config['num_classes']}")
    print(f"  Deep supervision: {config['deep_supervision']}")
    print(f"  Input size: {config['input_h']}x{config['input_w']}")
    
    return modified_model

def export_to_onnx(model, args, config):
    """将PyTorch模型导出为ONNX格式"""
    onnx_path = os.path.join(args.model_dir, args.engine_name.replace('.engine', '.onnx'))
    
    print(f"\nExporting model to ONNX: {onnx_path}")
    
    # 确定批处理大小
    batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    
    # 创建示例输入
    dummy_input = torch.randn(
        batch_size, 
        config['input_channels'], 
        config['input_h'], 
        config['input_w']
    )
    
    # 设置动态轴
    dynamic_axes = {}
    if args.dynamic_batch:
        dynamic_axes['input'] = {0: 'batch_size'}
        dynamic_axes['output'] = {0: 'batch_size'}
        if config['deep_supervision']:
            dynamic_axes['aux1'] = {0: 'batch_size'}
            dynamic_axes['aux2'] = {0: 'batch_size'}
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'] if not config['deep_supervision'] else ['aux1', 'aux2', 'output'],
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        verbose=args.verbose
    )
    
    print(f"ONNX model saved to {onnx_path}")
    return onnx_path

def build_engine_from_onnx(onnx_path, args, config):
    """从ONNX模型构建TensorRT引擎"""
    print("\nBuilding TensorRT engine from ONNX...")
    
    # 创建builder和config
    builder = trt.Builder(TRT_LOGGER)
    trt_config = builder.create_builder_config()
    
    # 设置工作空间大小
    trt_config.max_workspace_size = args.workspace_size * (1 << 30)  # GB to bytes
    
    # 设置精度模式
    if args.fp16:
        trt_config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    
    if args.int8:
        trt_config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision")
    
    # 解析ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f"Network inputs: {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")
    
    # 设置输入形状
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    
    # 确定批处理大小
    batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    
    if args.dynamic_batch:
        # 设置动态形状范围
        min_shape = (args.min_batch, config['input_channels'], config['input_h'], config['input_w'])
        opt_shape = (batch_size, config['input_channels'], config['input_h'], config['input_w'])
        max_shape = (args.max_batch, config['input_channels'], config['input_h'], config['input_w'])
    else:
        # 固定形状
        min_shape = (batch_size, config['input_channels'], config['input_h'], config['input_w'])
        opt_shape = min_shape
        max_shape = min_shape
    
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    trt_config.add_optimization_profile(profile)
    
    print(f"Input shape configuration:")
    print(f"  Min: {min_shape}")
    print(f"  Opt: {opt_shape}")
    print(f"  Max: {max_shape}")
    
    # 构建引擎
    print("\nBuilding TensorRT engine... This may take a while.")
    engine = builder.build_engine(network, trt_config)
    
    if engine is None:
        print("Failed to build engine")
        return None
    
    print("TensorRT engine built successfully")
    return engine

def save_engine(engine, engine_path):
    """保存TensorRT引擎到文件"""
    print(f"\nSaving TensorRT engine to {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"Engine saved successfully")

def test_engine(engine_path, args, config):
    """测试保存的引擎是否可以正常加载和推理"""
    print("\nTesting saved engine...")
    
    # 加载引擎
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("Failed to load engine")
        return
    
    # 创建执行上下文
    context = engine.create_execution_context()
    
    # 确定批处理大小
    batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    
    # 准备输入输出
    input_shape = (batch_size, config['input_channels'], config['input_h'], config['input_w'])
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 分配GPU内存
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # 计算内存大小
    input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
    d_input = cuda.mem_alloc(input_size)
    
    # 获取输出数量
    num_outputs = len([i for i in range(engine.num_bindings) if not engine.binding_is_input(i)])
    outputs = []
    d_outputs = []
    
    # 为每个输出分配内存
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            output_shape = tuple(context.get_binding_shape(i))
            output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
            d_output = cuda.mem_alloc(output_size)
            d_outputs.append(d_output)
            outputs.append(np.zeros(output_shape, dtype=np.float32))
    
    # 创建stream
    stream = cuda.Stream()
    
    # 拷贝输入到GPU
    cuda.memcpy_htod_async(d_input, dummy_input, stream)
    
    # 创建bindings列表
    bindings = [int(d_input)] + [int(d_out) for d_out in d_outputs]
    
    # 执行推理
    context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle
    )
    
    # 拷贝输出到CPU
    for i, (d_output, output) in enumerate(zip(d_outputs, outputs)):
        cuda.memcpy_dtoh_async(output, d_output, stream)
    
    stream.synchronize()
    
    print(f"\nTest inference successful!")
    print(f"Input shape: {input_shape}")
    print(f"Number of outputs: {num_outputs}")
    
    if config['deep_supervision'] and num_outputs > 1:
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
            print(f"Output {i} range: [{output.min():.4f}, {output.max():.4f}]")
    else:
        output = outputs[0]
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

def main():
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        TRT_LOGGER.min_severity = trt.Logger.INFO
    
    # 构建路径
    config_path = os.path.join(args.model_dir, 'config.yml')
    model_path = os.path.join(args.model_dir, args.model_name)
    engine_path = os.path.join(args.model_dir, args.engine_name)
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # 1. 加载配置文件
        config = load_config(config_path)
        
        # 2. 加载并转换 PyTorch 模型
        model = load_enet_model_with_conversion(model_path, config)
        
        # 3. 导出为ONNX
        onnx_path = export_to_onnx(model, args, config)
        
        # 4. 构建TensorRT引擎
        engine = build_engine_from_onnx(onnx_path, args, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return
        
        # 5. 保存引擎
        save_engine(engine, engine_path)
        
        # 6. 测试引擎
        test_engine(engine_path, args, config)
        
        # 7. 清理中间文件
        if not args.keep_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)
            print(f"\nRemoved intermediate ONNX file: {onnx_path}")
        
        # 8. 打印转换摘要
        print("\n" + "="*50)
        print("Conversion Summary:")
        print("="*50)
        print(f"Model: {config['name']}")
        print(f"Architecture: {config['arch']} (Modified for TensorRT)")
        print(f"Input size: {config['input_h']}x{config['input_w']}")
        print(f"Number of classes: {config['num_classes']}")
        print(f"Deep supervision: {config['deep_supervision']}")
        print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")
        print(f"Dynamic batch: {args.dynamic_batch}")
        if args.dynamic_batch:
            print(f"Batch range: [{args.min_batch}, {args.max_batch}]")
        else:
            batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
            print(f"Fixed batch size: {batch_size}")
        print(f"Engine saved to: {engine_path}")
        print("="*50)
        
        print("\nConversion completed successfully!")
        print("\n⚠️  Note: The model has been modified to replace MaxUnpool2d with bilinear upsampling")
        print("for TensorRT compatibility. This may slightly affect the accuracy.")
        print("Please validate the converted model's performance on your validation dataset.")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()