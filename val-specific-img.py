import argparse
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm
import numpy as np

import archs
from dataset import Dataset
from albumentations import Resize
from archs import UNext

# 导入FEDASNet
from model_zoo.fedas_net import FEDASNet

# python val-specific-img.py --name fedas_sperm_tail --input_image /path/to/input.jpg --output_folder /path/to/output/

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True, help='Model name')
    parser.add_argument('--input_image', required=True, help='Input image for prediction')
    parser.add_argument('--output_folder', required=True, help='Folder to save the predicted mask')
    parser.add_argument('--cuda', default='cuda:0', help='CUDA device')
    parser.add_argument('--save_prob', action='store_true', help='Save probability map')
    parser.add_argument('--save_features', action='store_true', help='Save intermediate features (FEDASNet only)')

    args = parser.parse_args()

    return args


def visualize_fedas_features(model, features, output_folder, img_id):
    """可视化FEDASNet的中间特征"""
    feature_folder = os.path.join(output_folder, 'features')
    os.makedirs(feature_folder, exist_ok=True)
    
    # 可视化区域图
    if 'region_map' in features:
        region_map = features['region_map']
        region_map = torch.sigmoid(region_map).cpu().numpy()[0, 0]
        region_map = (region_map * 255).astype('uint8')
        region_map_colored = cv2.applyColorMap(region_map, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(feature_folder, f'{img_id}_region_map.jpg'), region_map_colored)
    
    # 可视化注意力特征
    if 'attention_features' in features:
        attn_feat = features['attention_features']
        # 取第一个通道进行可视化
        attn_map = attn_feat[0, 0].cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map = (attn_map * 255).astype('uint8')
        attn_map_colored = cv2.applyColorMap(attn_map, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(feature_folder, f'{img_id}_attention_map.jpg'), attn_map_colored)


def main():
    args = parse_args()

    # 加载配置
    with open(f'models/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print(f'{key}: {config[key]}')
    print('-' * 20)

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # 创建模型
    print(f"=> creating model {config['arch']}")
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
    model.load_state_dict(torch.load(f'models/{args.name}/model.pth', map_location=device))
    model.eval()

    # 图像预处理
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 读取图像
    img_id = os.path.splitext(os.path.basename(args.input_image))[0]
    img = cv2.imread(args.input_image)
    if img is None:
        raise ValueError(f"Cannot read image: {args.input_image}")
    
    img_original = img.copy()  # 保存原始图像用于叠加显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 应用变换
    augmented = val_transform(image=img)
    img = augmented['image']
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(img)
        
        # 处理深度监督的情况
        if isinstance(output, list):
            # 保存所有深度监督的输出（可选）
            if config.get('deep_supervision', False):
                for idx, aux_output in enumerate(output[:-1]):
                    aux_prob = torch.sigmoid(aux_output).cpu().numpy()
                    aux_binary = (aux_prob >= 0.5).astype('uint8')
                    for c in range(config['num_classes']):
                        aux_folder = os.path.join(args.output_folder, f'aux_{idx}', str(c))
                        os.makedirs(aux_folder, exist_ok=True)
                        aux_resized = cv2.resize(aux_binary[0, c], (1920, 1200))
                        aux_resized = (aux_resized * 255).astype('uint8')
                        cv2.imwrite(os.path.join(aux_folder, f'{img_id}.jpg'), aux_resized)
            
            # 使用最后一个输出作为主要输出
            output = output[-1]
        
        # 获取概率图
        output_prob = torch.sigmoid(output).cpu().numpy()
        
        # 二值化
        output_binary = output_prob.copy()
        output_binary[output_binary >= 0.5] = 1
        output_binary[output_binary < 0.5] = 0

        # 如果是FEDASNet，可视化中间特征
        if config['arch'] == 'FEDASNet' and args.save_features:
            if hasattr(model, 'get_features'):
                features = model.get_features()
                visualize_fedas_features(model, features, args.output_folder, img_id)

        # 保存预测结果
        for c in range(config['num_classes']):
            # 创建输出目录
            os.makedirs(os.path.join(args.output_folder, str(c)), exist_ok=True)
            
            # 调整到原始尺寸
            output_resized = cv2.resize(output_binary[0, c], (1920, 1200))
            output_resized = (output_resized * 255).astype('uint8')
            
            # 创建彩色版本用于可视化
            output_colored = np.zeros((1200, 1920, 3), dtype=np.uint8)
            output_colored[:, :, 1] = output_resized  # 绿色通道显示预测结果
            
            # 查找轮廓并标记中心点
            contours, _ = cv2.findContours(output_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 在二值图上绘制轮廓和中心点
            output_with_centers = cv2.cvtColor(output_resized, cv2.COLOR_GRAY2BGR)
            
            center_points = []
            for contour in contours:
                # 计算轮廓面积，过滤噪声
                area = cv2.contourArea(contour)
                if area < 10:  # 过滤太小的轮廓
                    continue
                
                # 绘制轮廓
                cv2.drawContours(output_with_centers, [contour], -1, (0, 255, 0), 2)
                
                # 计算中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    center_points.append((cX, cY))
                    # 绘制中心点
                    cv2.circle(output_with_centers, (cX, cY), 5, (0, 0, 255), -1)
                    # 可选：添加编号
                    cv2.putText(output_with_centers, str(len(center_points)), 
                               (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 0), 1)
            
            # 保存二值掩码
            save_path = os.path.join(args.output_folder, str(c), f'{img_id}_mask.jpg')
            cv2.imwrite(save_path, output_resized)
            
            # 保存带中心点的结果
            save_path_centers = os.path.join(args.output_folder, str(c), f'{img_id}_centers.jpg')
            cv2.imwrite(save_path_centers, output_with_centers)
            
            # 保存概率图（如果需要）
            if args.save_prob:
                prob_resized = cv2.resize(output_prob[0, c], (1920, 1200))
                prob_resized = (prob_resized * 255).astype('uint8')
                prob_colored = cv2.applyColorMap(prob_resized, cv2.COLORMAP_JET)
                save_path_prob = os.path.join(args.output_folder, str(c), f'{img_id}_prob.jpg')
                cv2.imwrite(save_path_prob, prob_colored)
            
            # 创建叠加图像（原图+预测结果）
            if img_original.shape[:2] != (1200, 1920):
                img_original_resized = cv2.resize(img_original, (1920, 1200))
            else:
                img_original_resized = img_original
            
            # 创建半透明叠加
            overlay = img_original_resized.copy()
            mask_3channel = cv2.cvtColor(output_resized, cv2.COLOR_GRAY2BGR)
            mask_colored = np.zeros_like(mask_3channel)
            mask_colored[:, :, 1] = output_resized  # 绿色显示预测
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            # 在叠加图上标记中心点
            for cX, cY in center_points:
                cv2.circle(overlay, (cX, cY), 5, (0, 0, 255), -1)
            
            save_path_overlay = os.path.join(args.output_folder, str(c), f'{img_id}_overlay.jpg')
            cv2.imwrite(save_path_overlay, overlay)
            
            # 打印统计信息
            print(f"Class {c}:")
            print(f"  - Found {len(center_points)} objects")
            print(f"  - Total predicted pixels: {np.sum(output_binary[0, c])}")
            print(f"  - Coverage: {np.sum(output_binary[0, c]) / (output_binary[0, c].shape[0] * output_binary[0, c].shape[1]) * 100:.2f}%")
            
            # 保存中心点坐标到文本文件
            centers_file = os.path.join(args.output_folder, str(c), f'{img_id}_centers.txt')
            with open(centers_file, 'w') as f:
                f.write(f"Total objects: {len(center_points)}\n")
                f.write("Center coordinates (x, y):\n")
                for i, (cX, cY) in enumerate(center_points):
                    f.write(f"{i+1}: ({cX}, {cY})\n")

    print(f"\nResults saved to: {args.output_folder}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

'''
# 基本使用
python val-specific-img.py --name fedas_sperm_tail --input_image /path/to/input.jpg --output_folder /path/to/output/

# 保存概率图
python val-specific-img.py --name fedas_sperm_tail --input_image /path/to/input.jpg --output_folder /path/to/output/ --save_prob

# 保存FEDASNet的中间特征（仅对FEDASNet有效）
python val-specific-img.py --name fedas_sperm_tail --input_image /path/to/input.jpg --output_folder /path/to/output/ --save_features

# 指定CUDA设备
python val-specific-img.py --name fedas_sperm_tail --input_image /path/to/input.jpg --output_folder /path/to/output/ --cuda cuda:1
'''