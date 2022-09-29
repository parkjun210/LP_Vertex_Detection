from __future__ import annotations
import torch
import torchvision.transforms as transforms

import configparser
import argparse
import os
import sys
import cv2
from pathlib import Path
import numpy as np

from annotate.model import LPDetectionNet

SAVE_CWD = os.getcwd()
os.chdir(os.getcwd() + "/detection")
sys.path.append(os.getcwd())

from detection.execute import do_detect, build_detect_model

os.chdir(SAVE_CWD)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

class DetectLP:
    def __init__(self):
        self.args = None
        self.device = None
        self.detection_network = None
        self.transform = None

    def initialize(self, cfg_dir, useGPU=True):

        ################################################
        #   1. Read in parameters from config file     #
        ################################################
        if True:
            parser = argparse.ArgumentParser()

            args = parser.parse_args()

            config = configparser.RawConfigParser()
            config.read(cfg_dir)

            basic_config = config["basic_config"]

            # Weight Files
            args.detection_weight_file = basic_config["detection_weight_file"]
            if not os.path.exists(args.detection_weight_file):
                print(">>> NOT Exist DETECTION WEIGHT File {0}".format(args.detection_weight_file))
                #sys.exit(2)

            args.annotate_weight_file = basic_config["annotate_weight_file"]
            if not os.path.exists(args.detection_weight_file):
                print(">>> NOT Exist DETECTION WEIGHT File {0}".format(args.detection_weight_file))
                #sys.exit(2)

            # Input Data File
            args.source = basic_config["source"]
            if not os.path.exists(args.source):
                print(">>> NOT Exist INPUT File {0}".format(args.source))

            # GPU Number
            args.gpu_num = basic_config["gpu_num"]
            if args.gpu_num == "" :
                print(">>> NOT Assign GPU Number")
                #sys.exit(2)

            # Detection Parameters
            args.infer_imsize_same = basic_config.getboolean('infer_imsize_same')
            if args.infer_imsize_same == "" :
                args.infer_imsize_same = False

            args.detect_save_library = basic_config.getboolean('detect_save_library')
            if args.detect_save_library == "" :
                args.detect_save_library = False

            args.data = basic_config["data"]
            if args.data == "" :
                args.data = 'detection/data/AD.yaml'

            args.half = basic_config.getboolean('half')
            if args.half == "" :
                args.half = False

            imgsz = int(basic_config["detect_imgsz"])
            args.detect_imgsz = [imgsz]
            if args.detect_imgsz == "" :
                args.detect_imgsz = [640]

            args.conf_thres = float(basic_config["conf_thres"])
            if args.conf_thres == "" :
                args.conf_thres = 0.9

            args.iou_thres = float(basic_config["iou_thres"])
            if args.iou_thres == "" :
                args.iou_thres = 0.45

            args.max_det = int(basic_config["max_det"])
            if args.max_det == "" :
                args.max_det = 1000


            # Add Directory & Result save parameters
            args.output_dir = basic_config["output_dir"]
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            args.result_savefile = basic_config.getboolean('result_savefile')
            if args.result_savefile == "" :
                args.result_savefile = False

            args.save_detect_result = basic_config.getboolean('save_detect_result')
            if args.save_detect_result == "" :
                args.save_detect_result = False

            args.save_recog_result = basic_config.getboolean('save_recog_result')
            if args.save_recog_result == "" :
                args.save_recog_result = False

            args.hide_labels = basic_config.getboolean('hide_labels')
            if args.hide_labels == "" :
                args.hide_labels = False

            args.hide_conf = basic_config.getboolean('hide_conf')
            if args.hide_conf == "" :
                args.hide_conf = False

            args.save_conf = basic_config.getboolean('save_conf')
            if args.save_conf == "" :
                args.save_conf = False


            # Other parameters
            args.deidentified_type = basic_config["deidentified_type"]
            if args.deidentified_type == "" :
                args.deidentified_type = 2


        ################################################
        #                2. Set up GPU                 #
        ################################################
        if True:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
            device = torch.device("cuda:0" if useGPU else "cpu")

        ################################################
        #  3. Declare Detection / Recognition Network  #
        ################################################
        if True:
            # Detection Network
            args.detect_weights = args.detection_weight_file
            detection_network, imgsz, stride, names, pt = build_detect_model(args, device)
            annotate_network = LPDetectionNet(args)
            # Add network parameters to args
            args.pt = pt
            args.stride = stride
            args.imgsz = imgsz
            args.names = names  

        ################################################
        #    4. Load Detection / Recognition Network   #
        ################################################
        if True:
            annotate_checkpoint = torch.load(args.annotate_weight_file, map_location=device)
            annotate_network.load_state_dict(annotate_checkpoint['network'])            
            annotate_network.to(device)
            with torch.no_grad():
                detection_network.eval()
                annotate_network.eval()

        self.args = args
        self.device = device
        self.detection_network = detection_network
        self.annotate_network = annotate_network
        self.transform = transforms.ToTensor()

    def detect(self, img_tensor, img_mat):

        img_tensor = 255*img_tensor.permute(1,2,0)

        detect_preds = do_detect(self.args, self.detection_network, img_tensor, self.args.imgsz, self.args.stride, auto=True)

        return detect_preds

    def file_to_torchtensor(self, imgname):
        
        img_mat = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2RGB)

        img_tensor = self.mat_to_torchtensor(img_mat)

        return (img_mat, img_tensor)
    
    def mat_to_torchtensor(self, img_mat):
        
        img_tensor = self.transform(img_mat)
        img_tensor = img_tensor.to(self.device)

        return img_tensor

    def extend_bbox(self, bboxes, img_mat):

        H,W,_ = img_mat.shape

        ext_bboxes = bboxes.clone()

        for idx, bbox in enumerate(bboxes):
            cx = (bbox[2] + bbox[0])/2
            cy = (bbox[3] + bbox[1])/2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            new_x1 = max(cx - 0.75*w, 0)
            new_x2 = min(cx + 0.75*w, W)
            new_y1 = max(cy - 0.75*h, 0)
            new_y2 = min(cy + 0.75*h, H)

            ext_bboxes[idx][0] = int(new_x1)
            ext_bboxes[idx][1] = int(new_y1)
            ext_bboxes[idx][2] = int(new_x2)
            ext_bboxes[idx][3] = int(new_y2)
        
        return ext_bboxes

    def point_detect(self, img_mat, extend_bbox):

        transform_img = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize([int(128), int(256)]),
            ]
        )    

        extend_bbox = extend_bbox.detach().cpu().numpy()

        f_preds = []

        for bbox in extend_bbox:

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            img_crop = img_mat[y1:y2, x1:x2,:]

            img_crop_t = transform_img(img_crop)
            img_crop_t = img_crop_t.to(self.device)
            img_crop_t = torch.unsqueeze(img_crop_t, dim=0)

            pred = self.annotate_network(img_crop_t)

            pred = pred.detach().cpu().numpy().squeeze()

            h,w,c = img_crop.shape
            pred[0::2] = pred[0::2] * w
            pred[1::2] = pred[1::2] * h

            pred = pred.astype('int32')

            f_x1 = x1 + pred[0]
            f_x2 = x1 + pred[2]
            f_x3 = x1 + pred[4]
            f_x4 = x1 + pred[6]
            f_y1 = y1 + pred[1]
            f_y2 = y1 + pred[3]
            f_y3 = y1 + pred[5]
            f_y4 = y1 + pred[7]
            
            f_preds.append([f_x1, f_y1, f_x2, f_y2, f_x3, f_y3, f_x4, f_y4])

        return f_preds

    def swap_lp(self, img_mat, point_preds, ref_img):

        ref_img = cv2.cvtColor(cv2.imread(ref_img), cv2.COLOR_RGB2BGR)

        H, W, _ = img_mat.shape
        ref_img = cv2.resize(ref_img, (H,W))

        for point_pred in point_preds:

            point_matrix = np.float32([[0,0], [H, 0], [0,W], [H,W]])
            
            lu = [point_pred[0], point_pred[1]]
            ru = [point_pred[6], point_pred[7]]
            ld = [point_pred[2], point_pred[3]]
            rd = [point_pred[4], point_pred[5]]

            converted_points = np.float32([lu, ru, ld, rd])

            perspective_transform = cv2.getPerspectiveTransform(point_matrix,converted_points)

            warped = cv2.warpPerspective(ref_img,perspective_transform,(W,H))

            mask = (np.mean(warped, axis=2) > 0).astype(np.uint8)
            mask = np.stack([mask, mask, mask], axis=-1)

            img_mat = (1 - mask) * img_mat + mask * warped

        return img_mat

if __name__ == "__main__":
    
    detectlp = DetectLP()
    detectlp.initialize('detect.cfg', useGPU=True)
    img_mat, img_tensor = detectlp.file_to_torchtensor('dataset/images/img.png')
    bbox = detectlp.detect(img_tensor, img_mat)
    extend_bbox = detectlp.extend_bbox(bbox, img_mat)
    point_preds = detectlp.point_detect(img_mat, extend_bbox)
    img_swapped = detectlp.swap_lp(img_mat, point_preds, 'lp_reference.jpg')

    cv2.imwrite('swapped_result.jpg', cv2.cvtColor(img_swapped, cv2.COLOR_RGB2BGR))
