import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import Resize
from archs import UNext

# python val_single.py --name XY-local-UNeXt-all299-0804 --input_folder /223010087/SimonWorkspace/paper2/segmentation/UNeXt-pytorch/models/XY-local-UNeXt-First114-0804-cuda1/test-img/ --mask_folder models//XY-local-UNeXt-First114-0804-cuda1//test-img-mask//

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', required=True,
                        help='Model name')
    parser.add_argument('--input_folder', required=True, default="models//XY-local-UNeXt-First114-0804//test-img//",
                        help='Input folder containing images for prediction')
    parser.add_argument('--mask_folder', required=True, default="models//XY-local-UNeXt-First114-0804//test-img-mask//",
                        help='Mask folder containing images for prediction')
    
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

    # Data loading code
    img_ids = glob(os.path.join(args.input_folder, '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    model.load_state_dict(torch.load(f'models/{args.name}/model.pth'))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=img_ids,
        img_dir=args.input_folder,
        mask_dir=args.mask_folder,  # No masks needed for prediction
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

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], 'test-img', str(c)), exist_ok=True)
    
    with torch.no_grad():
        for input, _, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            model = model.cuda()

            # Compute output
            output = model(input)

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    output_resized = cv2.resize(output[i, c], (1920, 1200))  # Adjust image size to 1920x1200
                    output_resized = (output_resized * 255).astype('uint8')  # Convert image data type

                    save_path = os.path.join('outputs', config['name'], 'test-img', str(c), meta['img_id'][i] + '.jpg')
                    cv2.imwrite(save_path, output_resized)  # Save resized image

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()