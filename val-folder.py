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

# python val.py --name XY-local-UNeXt-all299-0804 --val_folder test_images

# 定义一个修改后的Dataset类，在没有掩码时创建虚拟掩码
class ModifiedDataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir  # 可能为None
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.has_mask = mask_dir is not None and os.path.exists(mask_dir)
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.has_mask:
            # 正常读取掩码
            mask = []
            for i in range(self.num_classes):
                mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
                if os.path.exists(mask_path):
                    mask.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None])
                else:
                    # 如果特定类别的掩码不存在，创建空掩码
                    mask.append(np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8))
            mask = np.dstack(mask)
        else:
            # 创建全黑掩码
            mask = np.zeros((img.shape[0], img.shape[1], self.num_classes), dtype=np.uint8)
            
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        img = img.astype('float32') / 255
        mask = mask.astype('float32') / 255
        
        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
            
        return img, mask, {'img_id': img_id}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--cuda', default="cuda:0",
                        help='cuda device')
    parser.add_argument('--val_folder', default=None,
                        help='path to validation images folder (if not specified, uses default dataset split)')
    parser.add_argument('--mask_folder', default=None,
                        help='path to validation masks folder (for computing metrics, optional)')
    parser.add_argument('--output_size', default='1600,1200',
                        help='output image size in format "width,height"')
    parser.add_argument('--img_ext', default=None,
                        help='image extension to use (e.g., .jpg, .png); if not specified, auto-detect in val_folder')

    args = parser.parse_args()

    return args

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

def verify_mask_folder(mask_dir, num_classes):
    """验证掩码文件夹是否存在且包含所需的子文件夹"""
    if mask_dir is None:
        return False
    
    if not os.path.exists(mask_dir):
        print(f"警告: 掩码文件夹 {mask_dir} 不存在")
        return False
    
    # 检查每个类别的子文件夹
    valid = True
    for c in range(num_classes):
        class_dir = os.path.join(mask_dir, str(c))
        if not os.path.exists(class_dir):
            print(f"警告: 类别 {c} 的掩码文件夹 {class_dir} 不存在")
            valid = False
            break
            
        # 检查文件夹是否为空
        if len(os.listdir(class_dir)) == 0:
            print(f"警告: 类别 {c} 的掩码文件夹 {class_dir} 为空")
            valid = False
            break
    
    return valid

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

    # 解析输出尺寸
    try:
        OUTPUT_WIDTH, OUTPUT_HEIGHT = map(int, args.output_size.split(','))
    except:
        # 默认尺寸
        OUTPUT_WIDTH, OUTPUT_HEIGHT = 1600, 1200
        print(f"使用默认输出尺寸: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")

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

    # Data loading code
    if args.val_folder:
        # 使用指定的验证文件夹，自动检测或使用指定的图像扩展名
        val_img_paths, detected_img_ext = get_image_files(args.val_folder, args.img_ext)
        print(f"检测到验证文件夹中的图像扩展名: {detected_img_ext}")
        
        val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_paths]
        img_dir = args.val_folder
        img_ext = detected_img_ext
        
        # 验证掩码文件夹是否有效
        mask_dir = args.mask_folder
        has_valid_masks = verify_mask_folder(mask_dir, config['num_classes'])
        if not has_valid_masks:
            print("将使用虚拟掩码进行预测（不计算评估指标）")
            mask_dir = None  # 设置为None以便后续逻辑知道不应该计算指标
        
        mask_ext = config['mask_ext']  # 使用配置文件中的mask扩展名
        
        # 设置输出目录为 outputs/val_folder
        output_base_dir = os.path.join('outputs', os.path.basename(args.val_folder.rstrip('/')))
    else:
        # 使用默认数据集的分割
        img_ext = config['img_ext']
        mask_ext = config['mask_ext']
        
        img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41, shuffle=False)
        img_dir = os.path.join('inputs', config['dataset'], 'images')
        mask_dir = os.path.join('inputs', config['dataset'], 'masks')
        has_valid_masks = verify_mask_folder(mask_dir, config['num_classes'])
        
        # 使用默认的输出目录
        output_base_dir = os.path.join('outputs', config['name'])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 创建数据集 - 使用修改后的数据集类
    val_dataset = ModifiedDataset(
        img_ids=val_img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir if has_valid_masks else None,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config['batch_size'],
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 评估指标
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    processing_times = []  # 存储每张图片的处理时间
    is_first_batch = True  # 标记是否是第一个批次（用于预热）

    # 创建输出目录
    for c in range(config['num_classes']):
        os.makedirs(os.path.join(output_base_dir, str(c)), exist_ok=True)
    
    # 评估循环
    print(f"开始评估 {len(val_img_ids)} 张图片...")
    print(f"输出将保存到: {output_base_dir}/")
    
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            
            # 开始计时 - 对模型前向传播和后处理进行计时
            start_time = time.time()
            
            # 前向传播
            output = model(input)
            
            # 处理深度监督的情况
            if isinstance(output, list):
                # 如果是深度监督，使用最后一个输出进行评估
                main_output = output[-1]
            else:
                main_output = output
            
            # 计算评估指标 (仅当有有效掩码时)
            if has_valid_masks:
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

            # 保存预测结果
            for i in range(len(output_binary)):
                for c in range(config['num_classes']):
                    # 调整图像大小到指定尺寸
                    output_resized = cv2.resize(output_binary[i, c], (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                    
                    # 转换数据类型为单通道 uint8 (0-255)
                    output_resized = (output_resized * 255).astype('uint8')

                    # 确保是单通道图像
                    if len(output_resized.shape) > 2:
                        output_resized = output_resized[:, :, 0]
                    
                    # 保存图像（不指定颜色映射，保持单通道）
                    save_path = os.path.join(output_base_dir, str(c), meta['img_id'][i] + '.png')
                    cv2.imwrite(save_path, output_resized)
                    
                    # 可选：同时保存概率图（用于后续分析）
                    if config.get('save_probability_maps', False):
                        prob_resized = cv2.resize(output_prob[i, c], (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                        prob_resized = (prob_resized * 255).astype('uint8')
                        
                        # 确保是单通道
                        if len(prob_resized.shape) > 2:
                            prob_resized = prob_resized[:, :, 0]
                            
                        prob_save_path = os.path.join(output_base_dir, str(c), meta['img_id'][i] + '_prob.png')
                        cv2.imwrite(prob_save_path, prob_resized)

    # 计算平均处理时间
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    total_processed_images = len(processing_times)

    # 打印最终结果
    print('='*50)
    print('Final Results:')
    if has_valid_masks:
        print('IoU: %.4f' % iou_avg_meter.avg)
        print('Dice: %.4f' % dice_avg_meter.avg)
    else:
        print('未提供有效的掩码文件夹，无法计算IoU和Dice指标')
    print('Output Size: %dx%d' % (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    print('Average Processing Time: %.4f ms per image (excluding warmup)' % avg_processing_time)
    print('Total Images Processed (excluding warmup): %d' % total_processed_images)
    print('='*50)
    
    # 如果是FEDASNet，可以打印更多信息
    if config['arch'] == 'FEDASNet':
        print('Model: FEDASNet')
        print('Lambda Fidelity: %.3f' % config.get('lambda_fidelity', 0.1))
        print('Lambda Region: %.3f' % config.get('lambda_region', 0.1))
        print('Lambda Boundary: %.3f' % config.get('lambda_boundary', 0.1))
    
    # 保存评估结果到文件
    result_file = os.path.join(output_base_dir, 'validation_results.txt')
    with open(result_file, 'w') as f:
        f.write('Model: %s\n' % config['arch'])
        if args.val_folder:
            f.write('Validation Folder: %s\n' % args.val_folder)
            f.write('Detected Image Extension: %s\n' % img_ext)
        else:
            f.write('Dataset: %s\n' % config['dataset'])
        
        if has_valid_masks:
            f.write('IoU: %.4f\n' % iou_avg_meter.avg)
            f.write('Dice: %.4f\n' % dice_avg_meter.avg)
        else:
            f.write('未提供有效的掩码文件夹，无法计算IoU和Dice指标\n')
            
        f.write('Output Size: %dx%d\n' % (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        f.write('Average Processing Time: %.4f ms per image (excluding warmup)\n' % avg_processing_time)
        f.write('Total Images Processed (excluding warmup): %d\n' % total_processed_images)
        if config['arch'] == 'FEDASNet':
            f.write('Lambda Fidelity: %.3f\n' % config.get('lambda_fidelity', 0.1))
            f.write('Lambda Region: %.3f\n' % config.get('lambda_region', 0.1))
            f.write('Lambda Boundary: %.3f\n' % config.get('lambda_boundary', 0.1))
    
    # 保存处理时间详情到一个单独的文件
    timing_file = os.path.join(output_base_dir, 'processing_times.txt')
    with open(timing_file, 'w') as f:
        f.write('Processing Time Details (excluding first batch for warmup):\n')
        f.write('Average Processing Time: %.4f ms per image\n' % avg_processing_time)
        f.write('Total Images Processed: %d\n' % total_processed_images)
        f.write('Total Processing Time: %.4f ms\n' % sum(processing_times))
        f.write('\nDetailed Processing Times (ms per image):\n')
        for i, t in enumerate(processing_times):
            f.write('Image %d: %.4f\n' % (i+1, t))

    # 复制配置文件到输出目录，便于追踪实验
    if args.val_folder:
        try:
            config_src = os.path.join('models', config['name'], 'config.yml')
            config_dst = os.path.join(output_base_dir, 'config.yml')
            shutil.copy2(config_src, config_dst)
            print(f"已复制配置文件到: {config_dst}")
        except Exception as e:
            print(f"复制配置文件时出错: {e}")

    torch.cuda.empty_cache()
    
    print(f"预测结果已保存到目录: {output_base_dir}/")


if __name__ == '__main__':
    main()