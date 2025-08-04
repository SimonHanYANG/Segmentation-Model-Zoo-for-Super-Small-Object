import argparse
import os
from glob import glob
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
from archs import UNext

# 导入FEDASNet
from model_zoo.fedas_net import FEDASNet
# 导入SpermEDANet
from model_zoo.spermedanet import SpermEDANet

# python val.py --name SpermEDANet_sperm_NAHeadLatest250610_0728

class BEAMHook:
    """专门用于BEAM模块的钩子函数类"""
    def __init__(self):
        self.attention_maps = {}
        self.input_tensors = {}
        self.current_batch_idx = 0
        self.edge_maps = {}
        self.channel_attentions = {}
        self.combined_attentions = {}
    
    def get_edge_map_hook(self, module_name):
        """获取边界注意力图的钩子函数"""
        def hook(module, input, output):
            # 保存边界注意力图（sigmoid激活后的输出）
            self.edge_maps[module_name] = output
        return hook
    
    def get_channel_att_hook(self, module_name):
        """获取通道注意力的钩子函数"""
        def hook(module, input, output):
            # 保存通道注意力
            self.channel_attentions[module_name] = output
        return hook
    
    def get_beam_input_hook(self, module_name):
        """获取BEAM模块输入的钩子函数"""
        def hook(module, input, output):
            # 保存模块的输入
            self.input_tensors[module_name] = input[0]
        return hook
    
    def compute_combined_attention(self):
        """计算组合注意力图"""
        for name in self.edge_maps:
            if name in self.channel_attentions and name in self.input_tensors:
                edge_map = self.edge_maps[name]
                channel_att = self.channel_attentions[name]
                # 计算组合注意力
                combined_att = channel_att * edge_map
                self.combined_attentions[name] = combined_att
    
    def get_all_attention_maps(self):
        """获取所有注意力图"""
        # 先计算组合注意力
        self.compute_combined_attention()
        
        # 收集所有注意力图
        result = {}
        for name in self.edge_maps:
            if name in self.edge_maps:
                result[f"{name}_edge_attention"] = self.edge_maps[name]
            if name in self.channel_attentions:
                result[f"{name}_channel_attention"] = self.channel_attentions[name]
            if name in self.combined_attentions:
                result[f"{name}_combined_attention"] = self.combined_attentions[name]
        
        return result
    
    def clear(self):
        """清除当前批次的注意力图"""
        self.edge_maps = {}
        self.channel_attentions = {}
        self.combined_attentions = {}
        self.input_tensors = {}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--cuda', default="cuda:0",
                        help='cuda device')
    parser.add_argument('--save-attention-maps', action='store_true', default=True,
                        help='save attention maps from SpermEDANet')

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

    # 创建模型 - 添加对SpermEDANet的支持
    print("=> creating model %s" % config['arch'])
    if config['arch'] == 'FEDASNet':
        model = FEDASNet(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            deep_supervision=config.get('deep_supervision', False)
        )
    elif config['arch'] == 'SpermEDANet':
        model = SpermEDANet(
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

    # 注册钩子获取中间特征和注意力图
    beam_hooks = BEAMHook()
    
    # 仅在SpermEDANet模型上注册钩子
    if config['arch'] == 'SpermEDANet':
        # 需要检查模型的具体结构和命名来注册正确的钩子
        for name, module in model.named_modules():
            # 为BEAM模块注册钩子
            if "stage2_beam" in name:
                # 注册BEAM模块的主钩子以获取输入
                module.register_forward_hook(beam_hooks.get_beam_input_hook(name))
                
                # 查找并注册边界注意力分支
                if hasattr(module, "edge_branch"):
                    # 为最后一层（sigmoid）注册钩子以获取边界注意力图
                    edge_branch = module.edge_branch
                    edge_last_layer = edge_branch[-1]  # 假设最后一层是sigmoid
                    edge_last_layer.register_forward_hook(beam_hooks.get_edge_map_hook(name))
                
                # 查找并注册通道注意力分支
                if hasattr(module, "fc"):
                    # 为fc的最后一层（sigmoid）注册钩子以获取通道注意力
                    fc = module.fc
                    fc_last_layer = fc[-1]  # 假设最后一层是sigmoid
                    fc_last_layer.register_forward_hook(beam_hooks.get_channel_att_hook(name))
        
        print("已注册 BEAM 模块的钩子")

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41, shuffle=False)

    # 加载模型权重
    model.load_state_dict(torch.load('models/%s/model.pth' % config['name'], map_location=device))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 评估指标
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    processing_times = []  # 存储每张图片的处理时间
    is_first_batch = True  # 标记是否是第一个批次（用于预热）

    # 创建输出目录
    output_base_dir = 'FeatureMapOutputs_SpermEDANet'
    for c in range(config['num_classes']):
        os.makedirs(os.path.join(output_base_dir, config['name'], str(c)), exist_ok=True)
    
    # 为注意力图创建目录
    if config['arch'] == 'SpermEDANet':
        attention_dir = os.path.join(output_base_dir, config['name'], 'attention_maps')
        os.makedirs(attention_dir, exist_ok=True)
        # 为不同类型的注意力图创建子目录
        os.makedirs(os.path.join(attention_dir, 'edge'), exist_ok=True)
        os.makedirs(os.path.join(attention_dir, 'channel'), exist_ok=True)
        os.makedirs(os.path.join(attention_dir, 'combined'), exist_ok=True)
        os.makedirs(os.path.join(attention_dir, 'overlay'), exist_ok=True)
    
    # 固定输出尺寸
    # For ROI segmentation
    OUTPUT_WIDTH = 1920
    OUTPUT_HEIGHT = 1200
    
    # 评估循环
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            
            # 开始计时 - 对模型前向传播和后处理进行计时
            start_time = time.time()
            
            # 清除上一批次的注意力图
            beam_hooks.clear()
            
            # 前向传播
            output = model(input)
            
            # 处理深度监督的情况
            if isinstance(output, list):
                # 如果是深度监督，使用最后一个输出进行评估
                main_output = output[-1]
            else:
                main_output = output
            
            # 计算评估指标
            iou, dice = iou_score(main_output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            # 后处理
            output_prob = torch.sigmoid(main_output).cpu().numpy()
            output_binary = output_prob.copy()
            output_binary[output_binary >= 0.5] = 1
            output_binary[output_binary < 0.5] = 0

            # 计算处理时间 (转换为毫秒)
            batch_processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 如果不是第一个批次（预热批次），则记录处理时间
            if not is_first_batch:
                # 对批次中的每张图片都记录相同的处理时间（批次处理时间/批次大小）
                per_image_time = batch_processing_time / input.size(0)
                processing_times.extend([per_image_time] * input.size(0))
            else:
                is_first_batch = False

            # 获取所有注意力图
            attention_maps = beam_hooks.get_all_attention_maps()
            
            # 打印一次注意力图的数量，用于调试
            if len(attention_maps) > 0 and not hasattr(main, 'printed_attention_count'):
                print(f"获取到 {len(attention_maps)} 个注意力图:")
                for k in attention_maps.keys():
                    print(f"  - {k}: 形状 {attention_maps[k].shape}")
                main.printed_attention_count = True

            # 保存预测结果
            for i in range(len(output_binary)):
                for c in range(config['num_classes']):
                    # 调整图像大小到固定尺寸
                    output_resized = cv2.resize(output_binary[i, c], (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                    
                    # 转换数据类型为单通道 uint8 (0-255)
                    output_resized = (output_resized * 255).astype('uint8')

                    # 确保是单通道图像
                    if len(output_resized.shape) > 2:
                        output_resized = output_resized[:, :, 0]
                    
                    # 保存图像（不指定颜色映射，保持单通道）
                    save_path = os.path.join(output_base_dir, config['name'], str(c), meta['img_id'][i] + '.png')
                    cv2.imwrite(save_path, output_resized)
                    
                    # 可选：同时保存概率图（用于后续分析）
                    if config.get('save_probability_maps', False):
                        prob_resized = cv2.resize(output_prob[i, c], (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                        prob_resized = (prob_resized * 255).astype('uint8')
                        
                        # 确保是单通道
                        if len(prob_resized.shape) > 2:
                            prob_resized = prob_resized[:, :, 0]
                            
                        prob_save_path = os.path.join(output_base_dir, config['name'], str(c), meta['img_id'][i] + '_prob.png')
                        cv2.imwrite(prob_save_path, prob_resized)
                
                # 保存注意力图（仅SpermEDANet模型）
                if config['arch'] == 'SpermEDANet' and attention_maps:
                    # 加载原始图像，用于叠加显示
                    orig_img_path = os.path.join('inputs', config['dataset'], 'images', meta['img_id'][i] + config['img_ext'])
                    orig_img = None
                    if os.path.exists(orig_img_path):
                        orig_img = cv2.imread(orig_img_path)
                        if orig_img is not None:
                            orig_img_resized = cv2.resize(orig_img, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                    
                    # 保存所有获取的注意力图
                    for att_name, att_tensor in attention_maps.items():
                        try:
                            # 确保批次索引在范围内
                            if i < att_tensor.shape[0]:
                                # 从批次中提取当前图像的注意力图
                                att_map = att_tensor[i].cpu().numpy()
                                
                                # 根据注意力图类型决定保存位置
                                att_type = 'generic'
                                if 'edge_attention' in att_name:
                                    att_type = 'edge'
                                elif 'channel_attention' in att_name:
                                    att_type = 'channel'
                                elif 'combined_attention' in att_name:
                                    att_type = 'combined'
                                
                                # 处理通道注意力（通常是 [C, 1, 1]）
                                if 'channel_attention' in att_name:
                                    # 通道注意力图通常是 [C, 1, 1] 或 [C, H, W]
                                    if att_map.ndim > 2 and att_map.shape[-1] == 1 and att_map.shape[-2] == 1:
                                        # 提取通道权重
                                        channel_weights = att_map.squeeze(-1).squeeze(-1)  # 形状: [C]
                                        
                                        # 保存通道权重为条形图
                                        plt_save_path = os.path.join(
                                            attention_dir, 'channel',
                                            f"{meta['img_id'][i]}_{att_name.split('/')[-1].replace('/', '_')}_weights.png"
                                        )
                                        
                                        plt.figure(figsize=(10, 4))
                                        plt.bar(range(len(channel_weights)), channel_weights)
                                        plt.title(f"Channel Attention Weights - {meta['img_id'][i]}")
                                        plt.xlabel("Channel Index")
                                        plt.ylabel("Attention Weight")
                                        plt.tight_layout()
                                        plt.savefig(plt_save_path)
                                        plt.close()
                                        
                                        # 由于是1x1，跳过热力图可视化
                                        continue
                                
                                # 处理边界注意力和组合注意力
                                # 确保是2D图像 - 移除多余维度
                                if att_map.ndim == 3 and att_map.shape[0] == 1:  # [1, H, W]
                                    att_map = att_map.squeeze(0)  # 变为 [H, W]
                                elif att_map.ndim > 3:  # 处理更复杂的情况
                                    # 如果是多通道的，但不是1x1的，取第一个通道
                                    if att_map.shape[0] > 1 and (att_map.shape[-1] > 1 or att_map.shape[-2] > 1):
                                        att_map = att_map[0]  # 取第一个通道
                                
                                # 调整大小以匹配输出尺寸
                                att_resized = cv2.resize(att_map, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                                
                                # 归一化到0-255范围并转换为uint8
                                att_min = att_resized.min()
                                att_max = att_resized.max()
                                if att_max > att_min:  # 避免除以零
                                    att_normalized = ((att_resized - att_min) / (att_max - att_min) * 255).astype(np.uint8)
                                else:
                                    att_normalized = np.zeros_like(att_resized, dtype=np.uint8)
                                
                                # 确保att_normalized是单通道uint8类型
                                if att_normalized.dtype != np.uint8:
                                    print(f"警告: 注意力图不是uint8类型，强制转换 ({att_normalized.dtype})")
                                    att_normalized = att_normalized.astype(np.uint8)
                                
                                # 确保是单通道
                                if len(att_normalized.shape) > 2:
                                    print(f"警告: 注意力图不是单通道，取第一个通道 ({att_normalized.shape})")
                                    att_normalized = att_normalized[:, :, 0]
                                
                                # 应用颜色映射以便于可视化
                                att_heatmap = cv2.applyColorMap(att_normalized, cv2.COLORMAP_JET)
                                
                                # 安全处理文件名 (避免路径问题)
                                safe_name = att_name.replace('/', '_').replace('\\', '_')
                                
                                # 保存热力图
                                att_save_path = os.path.join(
                                    attention_dir, att_type,
                                    f"{meta['img_id'][i]}_{safe_name}.png"
                                )
                                cv2.imwrite(att_save_path, att_heatmap)
                                
                                # 保存与原始图像的融合图
                                if orig_img is not None:
                                    # 创建注意力图与原始图像的融合
                                    alpha = 0.6  # 透明度
                                    overlay = cv2.addWeighted(orig_img_resized, 1 - alpha, att_heatmap, alpha, 0)
                                    
                                    # 保存融合图
                                    overlay_save_path = os.path.join(
                                        attention_dir, 'overlay',
                                        f"{meta['img_id'][i]}_{safe_name}_overlay.png"
                                    )
                                    cv2.imwrite(overlay_save_path, overlay)
                        except Exception as e:
                            print(f"处理注意力图 {att_name} 时出错: {str(e)}")
                            # 在出错时打印更详细的信息，帮助调试
                            print(f"注意力图形状: {att_tensor.shape}")
                            if i < att_tensor.shape[0]:
                                print(f"当前图像的注意力图形状: {att_tensor[i].shape}")
                                print(f"注意力图类型: {att_tensor.dtype}")
                                print(f"注意力图值范围: [{att_tensor[i].min().item()}, {att_tensor[i].max().item()}]")

    # 计算平均处理时间
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    total_processed_images = len(processing_times)

    # 打印最终结果
    print('='*50)
    print('Final Results:')
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('Output Size: %dx%d' % (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    print('Average Processing Time: %.4f ms per image (excluding warmup)' % avg_processing_time)
    print('Total Images Processed (excluding warmup): %d' % total_processed_images)
    print('='*50)
    
    # 如果是SpermEDANet，打印相关信息
    if config['arch'] == 'SpermEDANet':
        print('Model: SpermEDANet')
        print('注意力图已保存至: %s' % attention_dir)
    
    # 保存评估结果到文件
    result_file = os.path.join(output_base_dir, config['name'], 'validation_results.txt')
    with open(result_file, 'w') as f:
        f.write('Model: %s\n' % config['arch'])
        f.write('Dataset: %s\n' % config['dataset'])
        f.write('IoU: %.4f\n' % iou_avg_meter.avg)
        f.write('Dice: %.4f\n' % dice_avg_meter.avg)
        f.write('Output Size: %dx%d\n' % (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        f.write('Average Processing Time: %.4f ms per image (excluding warmup)\n' % avg_processing_time)
        f.write('Total Images Processed (excluding warmup): %d\n' % total_processed_images)
        if config['arch'] == 'SpermEDANet':
            f.write('注意力图保存位置: %s\n' % attention_dir)
    
    # 保存处理时间详情到一个单独的文件
    timing_file = os.path.join(output_base_dir, config['name'], 'processing_times.txt')
    with open(timing_file, 'w') as f:
        f.write('Processing Time Details (excluding first batch for warmup):\n')
        f.write('Average Processing Time: %.4f ms per image\n' % avg_processing_time)
        f.write('Total Images Processed: %d\n' % total_processed_images)
        f.write('Total Processing Time: %.4f ms\n' % sum(processing_times))
        f.write('\nDetailed Processing Times (ms per image):\n')
        for i, t in enumerate(processing_times):
            f.write('Image %d: %.4f\n' % (i+1, t))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()