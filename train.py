import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool, AverageMeter_CaraNet, clip_gradient
from archs import UNext

import FeiUnet
from FeiUnet import UNet

from model_zoo.fedas_net import FEDASNet
from model_zoo.caranet import CaraNet
from model_zoo.parnet import PraNet

ARCH_NAMES = archs.__all__ + ['FEDASNet']  # 添加FEDASNet
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default="cuda:0", type=str,
                        help='cuda:0 or cuda:1')
    
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=1920, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=1440, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # FEDAS specific parameters
    parser.add_argument('--lambda_fidelity', default=0.1, type=float,
                        help='weight for fidelity loss')
    parser.add_argument('--lambda_region', default=0.1, type=float,
                        help='weight for region loss')
    parser.add_argument('--lambda_boundary', default=0.1, type=float,
                        help='weight for boundary loss')
    
    # dataset
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'AdamW'],
                        help='optimizer')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate, 5e-5 for my FEDASNet')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 
                                'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def train_pranet(config, train_loader, model, criterion, optimizer):
    """PraNet专用训练函数"""
    avg_meters = {
        'loss': AverageMeter(),
        'lateral_5': AverageMeter(),
        'lateral_4': AverageMeter(),
        'lateral_3': AverageMeter(),
        'lateral_2': AverageMeter(),
        'iou': AverageMeter()
    }

    model.train()

    # 多尺度训练
    size_rates = [0.75, 1, 1.25]
    
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        for rate in size_rates:
            optimizer.zero_grad()
            
            input = input.to(config['cuda'])
            target = target.to(config['cuda'])
            
            # 调整输入尺寸
            trainsize = int(round(config['input_w']*rate/32)*32)
            if rate != 1:
                input = F.interpolate(input, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                target = F.interpolate(target, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # 前向传播
            if config['deep_supervision']:
                outputs = model(input)
                # 计算损失
                loss5 = criterion.structure_loss(outputs[0], target)
                loss4 = criterion.structure_loss(outputs[1], target)
                loss3 = criterion.structure_loss(outputs[2], target)
                loss2 = criterion.structure_loss(outputs[3], target)
                loss = loss5 + loss4 + loss3 + loss2
                
                # 计算IoU
                iou, dice = iou_score(outputs[3], target)  # 使用最终输出计算IoU
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)
                
                loss5 = torch.tensor(0.0).to(config['cuda'])
                loss4 = torch.tensor(0.0).to(config['cuda'])
                loss3 = torch.tensor(0.0).to(config['cuda'])
                loss2 = loss
                
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            clip_gradient(optimizer, 0.5)
            optimizer.step()
            
            # 只记录标准尺寸的损失
            if rate == 1:
                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['lateral_5'].update(loss5.item(), input.size(0))
                avg_meters['lateral_4'].update(loss4.item(), input.size(0))
                avg_meters['lateral_3'].update(loss3.item(), input.size(0))
                avg_meters['lateral_2'].update(loss2.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))
                
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('lat5', avg_meters['lateral_5'].avg),
            ('lat4', avg_meters['lateral_4'].avg),
            ('lat3', avg_meters['lateral_3'].avg),
            ('lat2', avg_meters['lateral_2'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('lateral_5', avg_meters['lateral_5'].avg),
        ('lateral_4', avg_meters['lateral_4'].avg),
        ('lateral_3', avg_meters['lateral_3'].avg),
        ('lateral_2', avg_meters['lateral_2'].avg),
        ('iou', avg_meters['iou'].avg)
    ])

def validate_pranet(config, val_loader, model, criterion):
    """PraNet专用验证函数"""
    avg_meters = {
        'loss': AverageMeter(),
        'lateral_5': AverageMeter(),
        'lateral_4': AverageMeter(),
        'lateral_3': AverageMeter(),
        'lateral_2': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter()
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(config['cuda'])
            target = target.to(config['cuda'])

            # 前向传播
            if config['deep_supervision']:
                outputs = model(input)
                # 计算损失
                loss5 = criterion.structure_loss(outputs[0], target)
                loss4 = criterion.structure_loss(outputs[1], target)
                loss3 = criterion.structure_loss(outputs[2], target)
                loss2 = criterion.structure_loss(outputs[3], target)
                loss = loss5 + loss4 + loss3 + loss2
                
                # 计算IoU
                iou, dice = iou_score(outputs[3], target)  # 使用最终输出计算IoU
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)
                
                loss5 = torch.tensor(0.0).to(config['cuda'])
                loss4 = torch.tensor(0.0).to(config['cuda'])
                loss3 = torch.tensor(0.0).to(config['cuda'])
                loss2 = loss

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['lateral_5'].update(loss5.item(), input.size(0))
            avg_meters['lateral_4'].update(loss4.item(), input.size(0))
            avg_meters['lateral_3'].update(loss3.item(), input.size(0))
            avg_meters['lateral_2'].update(loss2.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('lateral_5', avg_meters['lateral_5'].avg),
        ('lateral_4', avg_meters['lateral_4'].avg),
        ('lateral_3', avg_meters['lateral_3'].avg),
        ('lateral_2', avg_meters['lateral_2'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])



def train_caranet(config, train_loader, model, criterion, optimizer):
    """CaraNet专用训练函数"""
    avg_meters = {
        'loss': AverageMeter_CaraNet(),
        'lateral_5': AverageMeter_CaraNet(),
        'lateral_3': AverageMeter_CaraNet(),
        'lateral_2': AverageMeter_CaraNet(),
        'lateral_1': AverageMeter_CaraNet(),
        'iou': AverageMeter_CaraNet()
    }

    model.train()

    # 多尺度训练
    size_rates = [0.75, 1, 1.25]
    
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        for rate in size_rates:
            optimizer.zero_grad()
            
            input = input.to(config['cuda'])
            target = target.to(config['cuda'])
            
            # 调整输入尺寸
            trainsize = int(round(config['input_w']*rate/32)*32)
            if rate != 1:
                input = F.interpolate(input, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                target = F.interpolate(target, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # 前向传播
            if config['deep_supervision']:
                outputs = model(input)
                # 计算损失
                loss5 = criterion.structure_loss(outputs[0], target)
                loss3 = criterion.structure_loss(outputs[1], target)
                loss2 = criterion.structure_loss(outputs[2], target)
                loss1 = criterion.structure_loss(outputs[3], target)
                loss = loss5 + loss3 + loss2 + loss1
                
                # 计算IoU
                iou, dice = iou_score(outputs[0], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)
                
                loss5 = loss
                loss3 = torch.tensor(0.0).cuda()
                loss2 = torch.tensor(0.0).cuda()
                loss1 = torch.tensor(0.0).cuda()
                
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            clip_gradient(optimizer, 0.5)
            optimizer.step()
            
            # 只记录标准尺寸的损失
            if rate == 1:
                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['lateral_5'].update(loss5.item(), input.size(0))
                avg_meters['lateral_3'].update(loss3.item(), input.size(0))
                avg_meters['lateral_2'].update(loss2.item(), input.size(0))
                avg_meters['lateral_1'].update(loss1.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))
                
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('lat5', avg_meters['lateral_5'].avg),
            ('lat3', avg_meters['lateral_3'].avg),
            ('lat2', avg_meters['lateral_2'].avg),
            ('lat1', avg_meters['lateral_1'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('lateral_5', avg_meters['lateral_5'].avg),
        ('lateral_3', avg_meters['lateral_3'].avg),
        ('lateral_2', avg_meters['lateral_2'].avg),
        ('lateral_1', avg_meters['lateral_1'].avg),
        ('iou', avg_meters['iou'].avg)
    ])

def validate_caranet(config, val_loader, model, criterion):
    """CaraNet专用验证函数"""
    avg_meters = {
        'loss': AverageMeter_CaraNet(),
        'lateral_5': AverageMeter_CaraNet(),
        'lateral_3': AverageMeter_CaraNet(),
        'lateral_2': AverageMeter_CaraNet(),
        'lateral_1': AverageMeter_CaraNet(),
        'iou': AverageMeter_CaraNet(),
        'dice': AverageMeter_CaraNet()
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(config['cuda'])
            target = target.to(config['cuda'])

            # 前向传播
            if config['deep_supervision']:
                outputs = model(input)
                # 计算损失
                loss5 = criterion.structure_loss(outputs[0], target)
                loss3 = criterion.structure_loss(outputs[1], target)
                loss2 = criterion.structure_loss(outputs[2], target)
                loss1 = criterion.structure_loss(outputs[3], target)
                loss = loss5 + loss3 + loss2 + loss1
                
                # 计算IoU
                iou, dice = iou_score(outputs[0], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)
                
                loss5 = loss
                loss3 = torch.tensor(0.0).cuda()
                loss2 = torch.tensor(0.0).cuda()
                loss1 = torch.tensor(0.0).cuda()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['lateral_5'].update(loss5.item(), input.size(0))
            avg_meters['lateral_3'].update(loss3.item(), input.size(0))
            avg_meters['lateral_2'].update(loss2.item(), input.size(0))
            avg_meters['lateral_1'].update(loss1.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('lateral_5', avg_meters['lateral_5'].avg),
        ('lateral_3', avg_meters['lateral_3'].avg),
        ('lateral_2', avg_meters['lateral_2'].avg),
        ('lateral_1', avg_meters['lateral_1'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])


def train_fedas(config, train_loader, model, criterion, optimizer):
    """FEDASNet专用训练函数"""
    avg_meters = {
        'loss': AverageMeter(),
        'seg_loss': AverageMeter(),
        'fidelity_loss': AverageMeter(),
        'region_loss': AverageMeter(),
        'boundary_loss': AverageMeter(),
        'iou': AverageMeter()
    }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(config['cuda'])
        target = target.to(config['cuda'])

        # Forward pass with feature extraction
        # 为FEDASNet修改forward以返回中间特征
        output = model(input)
        
        # 获取中间特征（这需要修改模型以返回特征）
        features_dict = None
        if hasattr(model, 'get_features'):
            features_dict = model.get_features()
        
        # 计算损失
        loss_dict = criterion(output, target, features_dict)
        total_loss = loss_dict['total']
        
        # 计算IoU
        if isinstance(output, list):
            iou, dice = iou_score(output[-1], target)
        else:
            iou, dice = iou_score(output, target)

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 更新统计
        avg_meters['loss'].update(total_loss.item(), input.size(0))
        avg_meters['seg_loss'].update(loss_dict['seg'].item(), input.size(0))
        avg_meters['fidelity_loss'].update(loss_dict['fidelity'].item(), input.size(0))
        avg_meters['region_loss'].update(loss_dict['region'].item(), input.size(0))
        avg_meters['boundary_loss'].update(loss_dict['boundary'].item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('seg', avg_meters['seg_loss'].avg),
            ('fid', avg_meters['fidelity_loss'].avg),
            ('reg', avg_meters['region_loss'].avg),
            ('bnd', avg_meters['boundary_loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('seg_loss', avg_meters['seg_loss'].avg),
        ('fidelity_loss', avg_meters['fidelity_loss'].avg),
        ('region_loss', avg_meters['region_loss'].avg),
        ('boundary_loss', avg_meters['boundary_loss'].avg),
        ('iou', avg_meters['iou'].avg)
    ])


def train(config, train_loader, model, criterion, optimizer):
    """原始训练函数"""
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(config['cuda'])
        target = target.to(config['cuda'])

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                if config['arch'] == "UNet":
                    output = output["out"]
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        else:
            output = model(input)
            if config['arch'] == "UNet":
                output = output["out"]
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate_fedas(config, val_loader, model, criterion):
    """FEDASNet专用验证函数"""
    avg_meters = {
        'loss': AverageMeter(),
        'seg_loss': AverageMeter(),
        'fidelity_loss': AverageMeter(),
        'region_loss': AverageMeter(),
        'boundary_loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter()
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(config['cuda'])
            target = target.to(config['cuda'])

            # Forward pass
            output = model(input)
            
            features_dict = None
            if hasattr(model, 'get_features'):
                features_dict = model.get_features()
            
            # 计算损失
            loss_dict = criterion(output, target, features_dict)
            total_loss = loss_dict['total']
            
            # 计算IoU
            if isinstance(output, list):
                iou, dice = iou_score(output[-1], target)
            else:
                iou, dice = iou_score(output, target)

            # 更新统计
            avg_meters['loss'].update(total_loss.item(), input.size(0))
            avg_meters['seg_loss'].update(loss_dict['seg'].item(), input.size(0))
            avg_meters['fidelity_loss'].update(loss_dict['fidelity'].item(), input.size(0))
            avg_meters['region_loss'].update(loss_dict['region'].item(), input.size(0))
            avg_meters['boundary_loss'].update(loss_dict['boundary'].item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('seg_loss', avg_meters['seg_loss'].avg),
        ('fidelity_loss', avg_meters['fidelity_loss'].avg),
        ('region_loss', avg_meters['region_loss'].avg),
        ('boundary_loss', avg_meters['boundary_loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])


def validate(config, val_loader, model, criterion):
    """原始验证函数"""
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(config['cuda'])
            target = target.to(config['cuda'])

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                if config['arch'] == "UNet":
                    output = output["out"]
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                if config['arch'] == "UNet":
                    output = output["out"]
                loss = criterion(output, target)
                iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['arch'] == 'FEDASNet':
        # 使用FEDASNet专用损失
        criterion = losses.FEDASLoss(
            lambda_fidelity=config['lambda_fidelity'],
            lambda_region=config['lambda_region'],
            lambda_boundary=config['lambda_boundary']
        ).to(config['cuda'])
    elif config['loss'] == 'CaraNetLoss':
        criterion = losses.CaraNetLoss().to(config['cuda'])
    elif config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(config['cuda'])
    else:
        criterion = losses.__dict__[config['loss']]().to(config['cuda'])

    cudnn.benchmark = True

    # create model
    if config['arch'] == "UNet":
        model = FeiUnet.__dict__[config['arch']](num_classes=config['num_classes'])
    elif config['arch'] == "FEDASNet":
        model = FEDASNet(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            deep_supervision=config['deep_supervision']
        )
    elif config['arch'] == 'CaraNet':
        model = CaraNet(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            deep_supervision=config['deep_supervision']
        )
    elif config['arch'] == "PraNet":
        model = PraNet(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            deep_supervision=config['deep_supervision']
        )
    else:
        model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.to(config['cuda'])
    
    # get the number of models parameters
    print('Number of models parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('-' * 20)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], 
                                                   patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                           milestones=[int(e) for e in config['milestones'].split(',')], 
                                           gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41, shuffle=False)

    train_transform = Compose([
        RandomRotate90(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 根据模型选择训练和验证函数
    if config['arch'] == 'FEDASNet':
        train_fn = train_fedas
        validate_fn = validate_fedas
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('seg_loss', []),
            ('fidelity_loss', []),
            ('region_loss', []),
            ('boundary_loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_seg_loss', []),
            ('val_fidelity_loss', []),
            ('val_region_loss', []),
            ('val_boundary_loss', []),
            ('val_iou', []),
            ('val_dice', []),
        ])
    elif config['arch'] == 'CaraNet':
        train_fn = train_caranet
        validate_fn = validate_caranet
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('lateral_5', []),
            ('lateral_3', []),
            ('lateral_2', []),
            ('lateral_1', []),
            ('iou', []),
            ('val_loss', []),
            ('val_lateral_5', []),
            ('val_lateral_3', []),
            ('val_lateral_2', []),
            ('val_lateral_1', []),
            ('val_iou', []),
            ('val_dice', []),
        ])
    elif config['arch'] == 'PraNet':
        train_fn = train_pranet
        validate_fn = validate_pranet
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('lateral_5', []),
            ('lateral_4', []),
            ('lateral_3', []),
            ('lateral_2', []),
            ('iou', []),
            ('val_loss', []),
            ('val_lateral_5', []),
            ('val_lateral_4', []),
            ('val_lateral_3', []),
            ('val_lateral_2', []),
            ('val_iou', []),
            ('val_dice', []),
        ])
    else:
        train_fn = train
        validate_fn = validate
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_iou', []),
            ('val_dice', []),
        ])

    best_iou = 0
    trigger = 0
    
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))

        # train for one epoch
        train_log = train_fn(config, train_loader, model, criterion, optimizer)
        
        # evaluate on validation set
        val_log = validate_fn(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        # 打印日志
        if config['arch'] == 'FEDASNet':
            print('loss %.4f - seg %.4f - fid %.4f - reg %.4f - bnd %.4f - iou %.4f' % 
                  (train_log['loss'], train_log['seg_loss'], train_log['fidelity_loss'],
                   train_log['region_loss'], train_log['boundary_loss'], train_log['iou']))
            print('val_loss %.4f - val_seg %.4f - val_fid %.4f - val_reg %.4f - val_bnd %.4f - val_iou %.4f - val_dice %.4f' %
                  (val_log['loss'], val_log['seg_loss'], val_log['fidelity_loss'],
                   val_log['region_loss'], val_log['boundary_loss'], 
                   val_log['iou'], val_log['dice']))
        else:
            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
                  % (train_log['loss'], train_log['iou'], val_log['loss'], 
                     val_log['iou'], val_log['dice']))

        # 更新日志
        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        
        if config['arch'] == 'CaraNet':
            log['loss'].append(train_log['loss'])
            log['lateral_5'].append(train_log['lateral_5'])
            log['lateral_3'].append(train_log['lateral_3'])
            log['lateral_2'].append(train_log['lateral_2'])
            log['lateral_1'].append(train_log['lateral_1'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_lateral_5'].append(val_log['lateral_5'])
            log['val_lateral_3'].append(val_log['lateral_3'])
            log['val_lateral_2'].append(val_log['lateral_2'])
            log['val_lateral_1'].append(val_log['lateral_1'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])
        elif config['arch'] == 'FEDASNet':
            log['loss'].append(train_log['loss'])
            log['seg_loss'].append(train_log['seg_loss'])
            log['fidelity_loss'].append(train_log['fidelity_loss'])
            log['region_loss'].append(train_log['region_loss'])
            log['boundary_loss'].append(train_log['boundary_loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_seg_loss'].append(val_log['seg_loss'])
            log['val_fidelity_loss'].append(val_log['fidelity_loss'])
            log['val_region_loss'].append(val_log['region_loss'])
            log['val_boundary_loss'].append(val_log['boundary_loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])
        elif config['arch'] == 'PraNet':
            log['loss'].append(train_log['loss'])
            log['lateral_5'].append(train_log['lateral_5'])
            log['lateral_4'].append(train_log['lateral_4'])
            log['lateral_3'].append(train_log['lateral_3'])
            log['lateral_2'].append(train_log['lateral_2'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_lateral_5'].append(val_log['lateral_5'])
            log['val_lateral_4'].append(val_log['lateral_4'])
            log['val_lateral_3'].append(val_log['lateral_3'])
            log['val_lateral_2'].append(val_log['lateral_2'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])
        else:
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' % config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' % config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()