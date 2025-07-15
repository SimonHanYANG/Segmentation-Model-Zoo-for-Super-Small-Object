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

# python val-specific-img-local-pipette.py --name XY-local-UNeXt-all299-0804 --input_image /223010087/SimonWorkspace/paper2/segmentation/UNeXt-pytorch/models/XY-local-UNeXt-all299-0804/input-test.jpg --output_folder /223010087/SimonWorkspace/paper2/segmentation/UNeXt-pytorch/models/XY-local-UNeXt-all299-0804/predict-test-single-local-pipette/

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True, help='Model name')
    parser.add_argument('--input_image', required=True, help='Input image for prediction')
    parser.add_argument('--output_folder', required=True, help='Folder to save the predicted mask')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open(f'models/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print(f'{key}: {config[key]}')
    print('-' * 20)

    cudnn.benchmark = True

    print(f"=> creating model {config['arch']}")
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    model = model.cuda()
    model.load_state_dict(torch.load(f'models/{args.name}/model.pth'))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    img_id = os.path.splitext(os.path.basename(args.input_image))[0]
    img = cv2.imread(args.input_image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cannot change the channel info
    original_img = img.copy()  # Keep a copy of the original image for drawing

    # Apply transformations
    augmented = val_transform(image=img)
    img = augmented['image']
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img)

        output = torch.sigmoid(output).cpu().numpy()
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        for c in range(config['num_classes']):
            os.makedirs(os.path.join(args.output_folder, str(c)), exist_ok=True)
            output_resized = cv2.resize(output[0, c], (original_img.shape[1], original_img.shape[0]))
            output_resized = (output_resized * 255).astype('uint8')

            # Find contours and draw center points
            contours, _ = cv2.findContours(output_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(output_resized, (cX, cY), 5, (0, 0, 255), -1)
                    cv2.circle(original_img, (cX, cY), 5, (0, 0, 255), -1)
                    
                    # Calculate end points for the lines
                    angle_rad = np.pi / 7  # 4 for 45 degrees in radians
                    length = 150
                    # Right upper line
                    end_x1 = int(cX + length * np.cos(angle_rad))
                    end_y1 = int(cY - length * np.sin(angle_rad))
                    # Left upper line
                    end_x2 = int(cX - length * np.cos(angle_rad))
                    end_y2 = int(cY - length * np.sin(angle_rad))

                    cv2.line(original_img, (cX, cY), (end_x1, end_y1), (0, 0, 255), 2)
                    cv2.line(original_img, (cX, cY), (end_x2, end_y2), (0, 0, 255), 2)

            save_path = os.path.join(args.output_folder, str(c), img_id + '.jpg')
            cv2.imwrite(save_path, output_resized)
            cv2.imwrite(os.path.join(args.output_folder, str(c), img_id + '_original.jpg'), original_img)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()