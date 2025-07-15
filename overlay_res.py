import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def main():
    # 定义路径

    # entire images
    images_dir = 'inputs/Sperm_Selection_Video_Test/Basler acA1920-155ucMED (40214438)_20241105_115426118/'

    masks_root_dir = "outputs/"
    masks_name_dir = 'Basler acA1920-155ucMED (40214438)_20241105_115426118/'
    masks_base_dir = masks_root_dir + masks_name_dir


    # ROI images
    # images_dir = 'ROI-images/'
    # masks_base_dir = 'ADSCNet_sperm_ROINAHead_250707/'
    # masks_base_dir = 'CGNet_sperm_ROINAHead_250707/'
    # masks_base_dir = 'DDRNet_sperm_ROINAHead_250707/'
    # masks_base_dir = 'EDANet_sperm_ROINAHead_250707/'
    # masks_base_dir = 'ENet_sperm_ROINAHead_250707/'
    # masks_base_dir = 'sperm_ROINAHead_250707/'
    # masks_base_dir = 'ROI-GT-masks/'


    output_dir_name = 'HoveredResults/' + masks_name_dir
    output_dir = output_dir_name
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取0/文件夹中的所有mask文件名
    mask_files = glob(os.path.join(masks_base_dir, '0', '*.png'))
    
    # 如果没有找到文件，尝试其他常见图像扩展名
    if not mask_files:
        mask_files = glob(os.path.join(masks_base_dir, '0', '*.jpg'))
    if not mask_files:
        mask_files = glob(os.path.join(masks_base_dir, '0', '*.jpeg'))
    if not mask_files:
        mask_files = glob(os.path.join(masks_base_dir, '0', '*.tif'))
    
    # 检查是否找到mask文件
    if not mask_files:
        print(f"在{os.path.join(masks_base_dir, '0')}中未找到mask文件")
        return
    
    # 定义颜色映射（BGR顺序，因为OpenCV使用BGR）
    color_map = {
        '0': (0, 0, 255),    # 红色
        '1': (0, 255, 0),    # 绿色
        '2': (255, 0, 0),    # 蓝色
        '3': (0, 255, 255)   # 黄色
    }
    
    # 处理每一个mask文件
    for mask_file in tqdm(mask_files, desc="处理图像"):
        # 提取文件名（不带路径和扩展名）
        file_name = os.path.basename(mask_file)
        file_base_name = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1]
        
        # 查找原始图像
        image_file = find_file(images_dir, file_base_name)
        if not image_file:
            print(f"未找到对应的原始图像: {file_base_name}")
            continue
        
        # 读取原始图像
        original_image = cv2.imread(image_file)
        if original_image is None:
            print(f"无法读取原始图像: {image_file}")
            continue
        
        # 创建叠加图像
        overlay_image = original_image.copy()
        
        # 处理每个类别的mask
        for class_idx in ['0', '1', '2', '3']:
            # 构建mask文件路径
            class_mask_file = os.path.join(masks_base_dir, class_idx, file_name)
            
            # 如果当前类别的mask文件不存在，尝试查找不同扩展名的文件
            if not os.path.exists(class_mask_file):
                class_mask_file = find_file(os.path.join(masks_base_dir, class_idx), file_base_name)
            
            # 如果仍然找不到，跳过这个类别
            if not class_mask_file:
                print(f"类别 {class_idx} 的mask文件不存在: {file_base_name}")
                continue
            
            # 读取mask
            mask = cv2.imread(class_mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"无法读取mask: {class_mask_file}")
                continue
            
            # 确保mask和原始图像尺寸相同
            if mask.shape[:2] != original_image.shape[:2]:
                mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
            
            # 创建二值mask
            binary_mask = mask > 0
            
            # 创建彩色遮罩
            color_mask = np.zeros_like(original_image)
            color_mask[binary_mask] = color_map[class_idx]
            
            # 叠加到原始图像上
            cv2.addWeighted(color_mask, 0.5, overlay_image, 1, 0, overlay_image)
        
        # 保存结果
        output_file = os.path.join(output_dir, file_name)
        cv2.imwrite(output_file, overlay_image)
        # print(f"已保存叠加图像: {output_file}")

def find_file(directory, base_name):
    """在目录中查找与base_name匹配的文件，支持不同的扩展名"""
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
        file_path = os.path.join(directory, base_name + ext)
        if os.path.exists(file_path):
            return file_path
    return None

if __name__ == "__main__":
    main()