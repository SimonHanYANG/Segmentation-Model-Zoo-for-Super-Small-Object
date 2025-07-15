import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FEDASLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)
        return loss


class FEDASLoss(nn.Module):
    """FEDAS-Net的综合损失函数 - 数值稳定版本"""
    def __init__(self, lambda_fidelity=0.1, lambda_region=0.1, 
                 lambda_boundary=0.1, alpha_seg=0.5):
        super().__init__()
        # 降低辅助损失的权重，避免梯度爆炸
        self.lambda_fidelity = lambda_fidelity
        self.lambda_region = lambda_region
        self.lambda_boundary = lambda_boundary
        self.alpha_seg = alpha_seg
        
        # 基础分割损失组件
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
        # 区域损失组件
        self.balanced_bce = BalancedBCELoss()
        
        # 边界损失组件
        self.boundary_loss = BoundaryLoss()
        
        # 保真度损失组件
        self.structure_loss = StructureLoss()
        
    def forward(self, predictions, target, features_dict=None):
        """
        predictions: 网络输出 (可能是列表，如果使用深度监督)
        target: 真实标签
        features_dict: 包含中间特征的字典
        """
        # 添加数值检查
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("Warning: NaN or Inf in predictions")
            return self._get_safe_loss_dict()
        
        # 处理深度监督
        if isinstance(predictions, list):
            main_pred = predictions[-1]
            aux_preds = predictions[:-1]
        else:
            main_pred = predictions
            aux_preds = []
        
        # 裁剪预测值，避免极端值
        main_pred = torch.clamp(main_pred, min=-10, max=10)
        
        # 1. 基础分割损失
        try:
            seg_loss = self.alpha_seg * self.dice_loss(main_pred, target) + \
                       (1 - self.alpha_seg) * self.bce_loss(main_pred, target)
            
            # 检查seg_loss
            if torch.isnan(seg_loss) or torch.isinf(seg_loss):
                seg_loss = torch.tensor(1.0, device=main_pred.device)
        except Exception as e:
            print(f"Error in seg_loss: {e}")
            seg_loss = torch.tensor(1.0, device=main_pred.device)
        
        # 辅助损失
        aux_loss = 0
        for aux_pred in aux_preds:
            aux_pred = torch.clamp(aux_pred, min=-10, max=10)
            try:
                aux_loss += self.alpha_seg * self.dice_loss(aux_pred, target) + \
                            (1 - self.alpha_seg) * self.bce_loss(aux_pred, target)
            except:
                pass
        
        if aux_preds and aux_loss > 0:
            aux_loss /= len(aux_preds)
            seg_loss = 0.7 * seg_loss + 0.3 * aux_loss
        
        # 2. 保真度损失
        fidelity_loss = torch.tensor(0.0, device=main_pred.device)
        if features_dict is not None and 'encoder_features' in features_dict:
            try:
                fidelity_loss = self.structure_loss(features_dict['encoder_features'], target)
                if torch.isnan(fidelity_loss) or torch.isinf(fidelity_loss):
                    fidelity_loss = torch.tensor(0.0, device=main_pred.device)
            except Exception as e:
                print(f"Error in fidelity_loss: {e}")
                fidelity_loss = torch.tensor(0.0, device=main_pred.device)
        
        # 3. 区域损失
        region_loss = torch.tensor(0.0, device=main_pred.device)
        if features_dict is not None and 'region_map' in features_dict:
            try:
                region_map = features_dict['region_map']
                region_map = torch.clamp(region_map, min=-10, max=10)
                
                # 调整尺寸
                if region_map.shape[2:] != target.shape[2:]:
                    region_map = F.interpolate(region_map, size=target.shape[2:], 
                                             mode='bilinear', align_corners=True)
                
                region_target = self._generate_region_target(target)
                region_loss = self.balanced_bce(region_map, region_target)
                
                if torch.isnan(region_loss) or torch.isinf(region_loss):
                    region_loss = torch.tensor(0.0, device=main_pred.device)
            except Exception as e:
                print(f"Error in region_loss: {e}")
                region_loss = torch.tensor(0.0, device=main_pred.device)
        
        # 4. 边界损失
        boundary_loss = torch.tensor(0.0, device=main_pred.device)
        try:
            boundary_loss = self.boundary_loss(main_pred, target)
            if torch.isnan(boundary_loss) or torch.isinf(boundary_loss):
                boundary_loss = torch.tensor(0.0, device=main_pred.device)
        except Exception as e:
            print(f"Error in boundary_loss: {e}")
            boundary_loss = torch.tensor(0.0, device=main_pred.device)
        
        # 总损失
        total_loss = seg_loss + \
                     self.lambda_fidelity * fidelity_loss + \
                     self.lambda_region * region_loss + \
                     self.lambda_boundary * boundary_loss
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(1.0, device=main_pred.device)
        
        # 返回损失字典
        loss_dict = {
            'total': total_loss,
            'seg': seg_loss,
            'fidelity': fidelity_loss,
            'region': region_loss,
            'boundary': boundary_loss
        }
        
        return loss_dict
    
    def _get_safe_loss_dict(self):
        """返回安全的损失字典"""
        return {
            'total': torch.tensor(1.0),
            'seg': torch.tensor(1.0),
            'fidelity': torch.tensor(0.0),
            'region': torch.tensor(0.0),
            'boundary': torch.tensor(0.0)
        }
    
    def _generate_region_target(self, target):
        """生成区域标签（通过膨胀操作）"""
        kernel_size = 11  # 减小kernel size
        padding = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(target.device)
        kernel = kernel / kernel.sum()
        
        # 膨胀操作
        dilated = F.conv2d(target.float(), kernel, padding=padding)
        region_target = (dilated > 0.5).float()  # 使用阈值
        
        return region_target


class DiceLoss(nn.Module):
    """Dice损失 - 数值稳定版本"""
    def __init__(self, smooth=1.0):  # 增大smooth值
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # 使用torch.clamp避免sigmoid溢出
        pred = torch.sigmoid(torch.clamp(pred, min=-10, max=10))
        
        # 展平张量
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集
        intersection = (pred_flat * target_flat).sum()
        
        # Dice系数
        dice = (2. * intersection + self.smooth) / \
               (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class BalancedBCELoss(nn.Module):
    """平衡二元交叉熵损失 - 数值稳定版本"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # 裁剪预测值
        pred = torch.clamp(pred, min=-10, max=10)
        
        # 计算正负样本权重
        pos_count = (target == 1).float().sum() + 1.0  # 避免除零
        neg_count = (target == 0).float().sum() + 1.0
        total = pos_count + neg_count
        
        pos_weight = total / (2.0 * pos_count)
        pos_weight = torch.clamp(pos_weight, min=0.1, max=10.0)  # 限制权重范围
        
        # 使用reduction='mean'确保损失不会过大
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        return criterion(pred, target)


class BoundaryLoss(nn.Module):
    """边界损失 - 数值稳定版本"""
    def __init__(self, theta0=2, theta=3):  # 减小参数值
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        # 使用sigmoid并裁剪
        pred_sig = torch.sigmoid(torch.clamp(pred, min=-10, max=10))
        
        # 计算边界
        pred_boundary = self._compute_boundary(pred_sig)
        target_boundary = self._compute_boundary(target)
        
        # 使用简单的权重
        weights = 1.0 + (self.theta0 - 1.0) * target_boundary
        
        # 使用smooth L1损失代替MSE，更稳定
        boundary_loss = F.smooth_l1_loss(pred_boundary * weights, 
                                        target_boundary * weights,
                                        reduction='mean')
        
        return boundary_loss * 0.1  # 降低边界损失的权重
    
    def _compute_boundary(self, mask):
        """使用更稳定的边界检测"""
        # 使用更小的卷积核
        kernel = torch.tensor([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(mask.device)
        
        # 计算边界
        boundary = F.conv2d(mask, kernel, padding=1)
        boundary = torch.abs(boundary)
        
        # 归一化
        boundary = torch.clamp(boundary, min=0, max=1)
        
        return boundary


class StructureLoss(nn.Module):
    """结构保持损失 - 数值稳定版本"""
    def __init__(self):
        super().__init__()
        
    def forward(self, features, target):
        """
        features: 编码器特征列表
        target: 真实标签
        """
        total_loss = 0
        valid_features = 0
        
        for feat in features:
            try:
                # 归一化特征，避免梯度爆炸
                feat_norm = F.normalize(feat, dim=1)
                
                # 调整目标大小
                target_resized = F.interpolate(target, size=feat.shape[2:], 
                                             mode='nearest')
                
                # 使用简单的相似度损失
                target_expanded = target_resized.repeat(1, feat.shape[1], 1, 1)
                
                # 使用cosine相似度
                loss = 1 - F.cosine_similarity(feat_norm, target_expanded, dim=1).mean()
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss
                    valid_features += 1
            except Exception as e:
                print(f"Error in structure loss computation: {e}")
                continue
        
        if valid_features > 0:
            return total_loss / valid_features * 0.1  # 降低权重
        else:
            return torch.tensor(0.0).to(target.device)
        

# 添加到losses.py -- CaraNet Losses

class CaraNetLoss(nn.Module):
    def __init__(self):
        super(CaraNetLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, mask):
        # 处理深度监督的情况
        if isinstance(pred, list):
            loss = 0
            # 权重可以调整
            weights = [1.0, 0.8, 0.6, 0.4]
            for i, p in enumerate(pred):
                loss += weights[i] * self.structure_loss(p, mask)
            return loss
        else:
            return self.structure_loss(pred, mask)
    
    def structure_loss(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        
        return (wbce + wiou).mean()
    
class PraNetLoss(nn.Module):
    def __init__(self):
        super(PraNetLoss, self).__init__()
        
    def forward(self, pred, mask):
        # 处理深度监督的情况
        if isinstance(pred, list):
            loss = 0
            # 可以调整权重
            weights = [0.5, 0.3, 0.2, 1.0]  # 最后一个输出权重最大
            for i, p in enumerate(pred):
                loss += weights[i] * self.structure_loss(p, mask)
            return loss
        else:
            return self.structure_loss(pred, mask)
    
    def structure_loss(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        
        return (wbce + wiou).mean()