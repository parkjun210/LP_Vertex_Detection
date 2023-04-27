import os

import torch
from torch.utils.data import DataLoader

from config import parse_args
from model import LPDetectionNet
from utils.data_loaders import Visual_Dataset
from utils.helpers import *
from utils.eval import *

def visualize():
    
    args = parse_args()

    #### Argument ####
    
    GPU_NUM = 0
    # ckpt 파일 경로
    WEIGHT_PATH     = args.weights_path
    # data 폴더 안 inferece할 폴더 경로
    INFERENCE_DIR   = os.path.join('data', args.inference_dir)
    # inference 결과가 담길 폴더 경로
    SAVE_DIR        = os.path.join('result', args.inference_dir)
    # Inference result image 저장 여부
    SAVE_IMG_FLAG   = args.save_img_flag


    # Set up Dataset
    test_dataset = Visual_Dataset(args, INFERENCE_DIR)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Set up Network
    network = LPDetectionNet(args)

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the network to GPU if possible
    if torch.cuda.is_available():
        network.to(device)

    ckpt = torch.load(WEIGHT_PATH)
    network.load_state_dict(ckpt['network'])

    with torch.no_grad():
        inference(network, test_dataloader, device, save_dir = SAVE_DIR, save_img_flag = SAVE_IMG_FLAG)


if __name__ == '__main__':
    visualize()    
