import os
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from config import parse_args
from model import LPDetectionNet
from utils.data_loaders import Dataset
from utils.helpers import *
from utils.eval import *

def train():
    
    args = parse_args()

    GPU_NUM = args.gpu_num
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    EPOCHS = args.epochs
    PRINT_EVERY = args.print_every
    EVAL_EVERY = args.eval_every

    LOSS_TYPE = args.loss_type

    LR = args.lr

    EXP_DIR = os.path.join('experiments', args.exp_name)
    WEIGHTS = args.weights

    # Check if directory does not exist
    os.makedirs('experiments/', exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    # Set up logger
    filename = os.path.join(EXP_DIR, 'logs.txt')
    logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))
        logging.info('\t%15s:\t%s' % (key, value))

    # Set up Dataset
    train_dataset = Dataset(args, 'train')
    test_dataset = Dataset(args, 'test')

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # Set up Network
    network = LPDetectionNet(args)

    logging.info('Network Parameters : %.1f M' % (count_parameters(network) * 10**(-6)))

    # Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        network.to(device)

    # Load the pretrained model if exists
    init_epoch = 0
    best_metric = 0.0

    if os.path.exists(os.path.join(EXP_DIR, WEIGHTS)):
        logging.info('Recovering from %s ...' % os.path.join(EXP_DIR, WEIGHTS))
        checkpoint = torch.load(os.path.join(EXP_DIR, WEIGHTS))
        init_epoch = checkpoint['epoch_idx']
        best_metric = checkpoint['best_metric']
        network.load_state_dict(checkpoint['network'])
        logging.info('Recover completed. Current epoch = #%d' % (init_epoch))

    # Criterion
    if LOSS_TYPE == 'l2':
        criterion = nn.MSELoss()
    elif LOSS_TYPE == 'l1':
        criterion = nn.L1Loss()
    elif LOSS_TYPE == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # Create Optimizer / Scheduler
    optimizer = optim.AdamW(network.parameters(), lr=LR)

    # Check trainable parameters
    trainable_list, module_param_list, total_params = count_module_parameters(network)
    logging.info("********** Trainable Parameters **********")
    for idx in range(len(trainable_list)):
        logging.info("\t%15s : %.1f M" % (trainable_list[idx], module_param_list[idx] * 10**(-6)))

    logging.info("\t%15s : %.1f M" % ('Total',total_params * 10**(-6)))    

    # Train
    for epoch_idx in range(init_epoch+1, EPOCHS):

        network.train()

        train_loss = 0

        # Iterate over dataset
        for i, data in enumerate(tqdm(train_dataloader, desc=f"epoch {epoch_idx}")):
            
            img, label = data
            [img, label] = img2cuda([img, label], device)

            pred = network(img)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * BATCH_SIZE

        train_loss /= len(train_dataset)

        if epoch_idx % PRINT_EVERY == 0:
            logging.info('Epoch [%d/%d] Loss = %.5f LR = %.7f' % (epoch_idx, EPOCHS, train_loss, LR))

        if epoch_idx % EVAL_EVERY == 0:
            eval_metric = evaluate(network, test_dataloader, device, logging)    

            if eval_metric > best_metric:
                best_metric = eval_metric

                # Save Network
                save_path = os.path.join(EXP_DIR, WEIGHTS)
                torch.save({
                    'epoch_idx': epoch_idx,
                    'best_metric': best_metric,
                    'network' : network.state_dict(),
                }, save_path)

                logging.info('Saved checkpoint to %s ...' % save_path)

            logging.info('EVAL IOU = %.2f  BEST IOU = %.2f' % (eval_metric, best_metric))                

if __name__ == '__main__':
    train()    