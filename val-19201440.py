import argparse
import os
from glob import glob

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
import time
from archs import UNext

# 导入FEDASNet
from model_zoo.fedas_net import FEDASNet

# python val.py --name XY-local-UNeXt-all299-0804

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--cuda', default="cuda:0",
                        help='cuda device')

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
    gput = AverageMeter()
    cput = AverageMeter()

    # 创建输出目录
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
    # 评估循环
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            
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

            # 保存预测结果
            for i in range(len(output_binary)):
                for c in range(config['num_classes']):
                    # 调整图像大小到原始尺寸
                    if 'input_w' in config and 'input_h' in config:
                        # 如果配置中有输入尺寸，先调整到配置尺寸
                        output_resized = cv2.resize(output_binary[i, c], (config['input_w'], config['input_h']))
                    else:
                        # 否则直接调整到默认尺寸
                        output_resized = cv2.resize(output_binary[i, c], (1920, 1200))
                    
                    # 转换数据类型
                    output_resized = (output_resized * 255).astype('uint8')

                    # 保存图像
                    save_path = os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg')
                    cv2.imwrite(save_path, output_resized)
                    
                    # 可选：同时保存概率图（用于后续分析）
                    if config['arch'] == 'FEDASNet':
                        prob_resized = cv2.resize(output_prob[i, c], (config.get('input_w', 1920), config.get('input_h', 1200)))
                        prob_resized = (prob_resized * 255).astype('uint8')
                        prob_save_path = os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '_prob.jpg')
                        cv2.imwrite(prob_save_path, prob_resized)

    # 打印最终结果
    print('='*50)
    print('Final Results:')
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('='*50)
    
    # 如果是FEDASNet，可以打印更多信息
    if config['arch'] == 'FEDASNet':
        print('Model: FEDASNet')
        print('Lambda Fidelity: %.3f' % config.get('lambda_fidelity', 0.1))
        print('Lambda Region: %.3f' % config.get('lambda_region', 0.1))
        print('Lambda Boundary: %.3f' % config.get('lambda_boundary', 0.1))
    
    # 保存评估结果到文件
    result_file = os.path.join('outputs', config['name'], 'validation_results.txt')
    with open(result_file, 'w') as f:
        f.write('Model: %s\n' % config['arch'])
        f.write('Dataset: %s\n' % config['dataset'])
        f.write('IoU: %.4f\n' % iou_avg_meter.avg)
        f.write('Dice: %.4f\n' % dice_avg_meter.avg)
        if config['arch'] == 'FEDASNet':
            f.write('Lambda Fidelity: %.3f\n' % config.get('lambda_fidelity', 0.1))
            f.write('Lambda Region: %.3f\n' % config.get('lambda_region', 0.1))
            f.write('Lambda Boundary: %.3f\n' % config.get('lambda_boundary', 0.1))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()