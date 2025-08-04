import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision.utils import save_image

from models.generator.generator import Generator
from datasets.dataset import create_image_dataset
from options.test_options import TestOptions
from utils.misc import sample_data, postprocess




if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print('Cuda is available')
        cudnn.enabled = True
        cudnn.benchmark = True

    # Parse options
    opts = TestOptions().parse

    # Setup paths
    os.makedirs(opts.result_root, exist_ok=True)
    # opts.pre_trained = 'snapshots/ckpt_celeba/030000.pt'
    # opts.image_root = 'image-radar'
    # opts.mask_root = 'mask_radar'
    # opts.result_root = 'result'

    # Load model
    generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
    if opts.pre_trained:
        generator.load_state_dict(torch.load(opts.pre_trained)['generator'])
    else:
        print('Please provide pre-trained model!')

    if is_cuda:
        generator = generator.cuda()

    # Create dataset and data loader
    image_dataset = create_image_dataset(opts)
    image_data_loader = data.DataLoader(
        image_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        drop_last=False
    )
    image_data_loader = sample_data(image_data_loader)

    # Collect all image file names from the dataset
    image_file_names = [img_name for img_name in os.listdir(opts.image_root) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    print('Start test...')
    with torch.no_grad():
        generator.eval()
        for i, (ground_truth, mask, edge, gray_image) in enumerate(image_data_loader):
            # Assuming the dataset returns the file name with each batch
            image_name = image_file_names[i]

            if is_cuda:
                ground_truth, mask, edge, gray_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda()

            input_image = ground_truth * mask
            input_edge = edge * mask
            input_gray_image = gray_image * mask

            output, _, _ = generator(input_image, torch.cat((input_edge, input_gray_image), dim=1), mask)
            output_comp = ground_truth * mask + output * (1 - mask)
            output_comp = postprocess(output_comp)

            # Save the output image with the same name as the input image
            output_image_path = os.path.join(opts.result_root, image_name)
            save_image(output_comp, output_image_path)
            print(f"图像 {image_name} 已处理并保存到 {output_image_path}")

            # Optional: Stop after processing the specified images
            if i == len(image_file_names) - 1:
                break
