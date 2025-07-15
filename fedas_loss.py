import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FEDASLoss']

class FEDASLoss(nn.Module):
    """FEDAS-Net的综合损失函数"""
    def __init__(self, lambda1=0.3, lambda2=0.2, lambda3=0.3, alpha=0.6, 
                 gamma=0.5, beta=0.4, delta=0.3):
        super().__init__()
        self.lambda1 = lambda1  # 保真度损失权重
        self.lambda2 = lambda2  # 区域损失权重  
        self.lambda3 = lambda3  # 边界损失权重
        self.alpha = alpha      # Dice与CE损失平衡
        self.gamma = gamma      # 方向一致性损失权重
        self.beta = beta        # 区域连贯性损失权重
        self.delta = delta      # 拓扑一致性损失权重
        
    def forward(self, outputs, targets):
        """
        outputs: 网络输出字典或张量
        targets: 真实标签
        """
        # 处理输出格式
        if isinstance(outputs, dict):
            pred = outputs['out']
            # 使用字典中的其他输出进行额外损失计算
            use_extra_losses = True
        else:
            pred = outputs
            use_extra_losses = False
            
        # 基础分割损失
        seg_loss = self.segmentation_loss(pred, targets)
        
        if use_extra_losses:
            # 保真度增强损失
            fidelity_loss = self.fidelity_enhancement_loss(
                outputs.get('hr_feat'), outputs.get('mr_feat'), 
                outputs.get('lr_feat'), targets
            )
            
            # 区域感知损失
            region_loss = self.region_aware_loss(
                outputs.get('region_map', pred), targets
            )
            
            # 边界增强损失
            boundary_loss = self.boundary_enhancement_loss(
                pred, outputs.get('boundary', pred), targets
            )
            
            # 总损失
            total_loss = seg_loss + \
                         self.lambda1 * fidelity_loss + \
                         self.lambda2 * region_loss + \
                         self.lambda3 * boundary_loss
        else:
            total_loss = seg_loss
            
        return total_loss
    
    def segmentation_loss(self, pred, target):
        """基础分割损失"""
        # Dice损失
        dice_loss = self.dice_loss(pred, target)
        
        # 二元交叉熵损失
        bce_loss = F.binary_cross_entropy(pred, target)
        
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss
    
    def dice_loss(self, pred, target, epsilon=1e-6):
        """Dice损失"""
        intersection = (pred * target).sum(dim=(2, 3))
        pred_sum = pred.pow(2).sum(dim=(2, 3))
        target_sum = target.pow(2).sum(dim=(2, 3))
        
        dice = 1 - (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)
        return dice.mean()
    
    def fidelity_enhancement_loss(self, hr_feat, mr_feat, lr_feat, target):
        """保真度增强损失"""
        if hr_feat is None:
            return torch.tensor(0.0, device=target.device)
            
        # 方向一致性损失
        direction_loss = self.direction_consistency_loss(hr_feat, target)
        
        return self.gamma * direction_loss
    
    def direction_consistency_loss(self, feat, target):
        """方向一致性损失"""
        # 将目标下采样到与特征相同的尺寸
        feat_h, feat_w = feat.shape[2:]
        target_resized = F.interpolate(target, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
        
        # 计算特征梯度
        if feat.shape[3] > 1:  # 检查宽度
            feat_grad_x = feat[:, :, :, 1:] - feat[:, :, :, :-1]
            target_grad_x = target_resized[:, :, :, 1:] - target_resized[:, :, :, :-1]
            # 计算x方向的方向一致性
            cos_sim_x = F.cosine_similarity(feat_grad_x.mean(dim=1, keepdim=True), 
                                            target_grad_x, dim=1)
            loss_x = 1 - cos_sim_x.mean()
        else:
            loss_x = 0
            
        if feat.shape[2] > 1:  # 检查高度
            feat_grad_y = feat[:, :, 1:, :] - feat[:, :, :-1, :]
            target_grad_y = target_resized[:, :, 1:, :] - target_resized[:, :, :-1, :]
            # 计算y方向的方向一致性
            cos_sim_y = F.cosine_similarity(feat_grad_y.mean(dim=1, keepdim=True), 
                                            target_grad_y, dim=1)
            loss_y = 1 - cos_sim_y.mean()
        else:
            loss_y = 0
        
        return loss_x + loss_y
    
    def region_aware_loss(self, region_map, target):
        """区域感知损失"""
        # 确保region_map和target尺寸一致
        if region_map.shape[2:] != target.shape[2:]:
            region_map = F.interpolate(region_map, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        # 平衡交叉熵损失
        pos_weight = (target.numel() - target.sum()) / (target.sum() + 1e-6)
        bce_loss = F.binary_cross_entropy(region_map, target)
        
        # 区域连贯性损失
        coherence_loss = self.region_coherence_loss(region_map, target)
        
        return bce_loss + self.beta * coherence_loss
    
    def region_coherence_loss(self, pred, target):
        """区域连贯性损失"""
        # 计算相邻像素的差异
        loss = 0
        count = 0
        
        if pred.shape[3] > 1:  # 检查宽度
            diff_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            edge_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            loss_x = diff_x * (1 - edge_x)
            loss += loss_x.mean()
            count += 1
            
        if pred.shape[2] > 1:  # 检查高度
            diff_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            edge_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
            loss_y = diff_y * (1 - edge_y)
            loss += loss_y.mean()
            count += 1
        
        return loss / max(count, 1)
    
    def boundary_enhancement_loss(self, pred, boundary, target):
        """边界增强损失"""
        # 确保pred和target尺寸一致
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        if boundary.shape[2:] != target.shape[2:]:
            boundary = F.interpolate(boundary, size=target.shape[2:], mode='bilinear', align_corners=False)
            
        # 计算距离变换权重
        weight_map = self.compute_boundary_weight(target)
        
        # 边界加权损失
        bw_loss = F.binary_cross_entropy(pred, target, weight=weight_map, reduction='mean')
        
        # 拓扑一致性损失
        topo_loss = self.topology_consistency_loss(pred, target)
        
        return bw_loss + self.delta * topo_loss
    
    def compute_boundary_weight(self, target, mu=5.0):
        """计算边界权重图"""
        # 获取设备
        device = target.device
        
        # 创建卷积核
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size * kernel_size)
        
        # 计算边界
        # 使用平均池化来近似膨胀操作
        dilated = F.conv2d(target, kernel, padding=1)
        dilated = (dilated > 0.1).float()
        
        # 使用1-target的平均池化来近似腐蚀操作
        eroded = F.conv2d(1 - target, kernel, padding=1)
        eroded = 1 - (eroded > 0.1).float()
        
        # 边界区域
        boundary = dilated - eroded
        boundary = boundary.clamp(0, 1)
        
        # 计算权重
        weight = 1 + mu * boundary
        
        return weight
    
    def topology_consistency_loss(self, pred, target):
        """拓扑一致性损失"""
        # 二值化预测
        pred_binary = (pred > 0.5).float()
        
        # 创建平滑核
        kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9
        
        # 计算连通性
        if pred_binary.shape[2] > 2 and pred_binary.shape[3] > 2:
            pred_smooth = F.conv2d(pred_binary, kernel, padding=1)
            target_smooth = F.conv2d(target, kernel, padding=1)
            
            # 连通性差异
            connectivity_diff = torch.abs(pred_smooth - target_smooth)
            
            return connectivity_diff.mean()
        else:
            return torch.tensor(0.0, device=pred.device)