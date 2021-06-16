import configparser
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import pickle
import json
import detectron2
from detectron2.utils.logger import setup_logger
import matplotlib.pyplot as plt
import pandas as pd
import sys
import show_predict_result
from detectron2.utils.visualizer import _create_text_labels, VisImage, GenericMask

# setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import datetime
import time
import os

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import uuid
from detectron2.utils.visualizer import ColorMode
import pickle

from PIL import Image

import pprint


class ConfigManager:
    def __init__(self):
        self.config = configparser.ConfigParser()

    def load(self, path: str):
        self.config.read(path)

        return self.config

    def dump(self, obj: dict, path: str):
        for key, value in obj.items():
            self.config[key] = value

        with open(path, 'w') as configfile:
            self.config.write(configfile)

        self.config = configparser.ConfigParser()


manager = ConfigManager()
config_dict = manager.load('/mnt/nfs_share/Xunjie/soil_cover.ini')


def predict_and_output(img):
    model_path = str(config_dict['path']['model_path'])

    # register_coco_instances("custom", {},
    #                         "/mnt/nfs_share/Xunjie/Dataset/0607/test/soil_cover_0607/COCO_segmentation.json",
    #                         "/mnt/nfs_share/Xunjie/Dataset/0607/test/soil_cover_0607/soil_cover_0607/")
    # custom_metadata = MetadataCatalog.get("custom")
    # DatasetCatalog.get("custom")

    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/xunjie/Downloads/fasterrcnn/3090_Test/detectron2/configs/COCO-InstanceSegmentation"
        "/mask_rcnn_R_50_FPN_1x.yaml")
    # cfg.DATASETS.TEST = ("custom",)
    cfg.MODEL.WEIGHTS = os.path.join(model_path, str(config_dict['predict']['model_name']))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(config_dict['predict']['SCORE_THRESH_TEST'])
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        int(config_dict['predict']['batch_size_per_image'])
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(config_dict['parameters']['num_classes'])
    predictor = DefaultPredictor(cfg)

    instance = predictor(img)["instances"].to("cpu")

    scores = instance.scores if instance.has("scores") else None
    classes = instance.pred_classes if instance.has("pred_classes") else None
    thing_classes = ['concrete', 'steel plate', 'green network', 'soil']
    labels = _create_text_labels(classes, scores, thing_classes)
    output_info = VisImage(img[:, :, ::-1], scale=1)

    if instance.has("pred_masks"):
        masks = np.asarray(instance.pred_masks)
        masks = [GenericMask(x, output_info.height, output_info.width) for x in masks]
    else:
        masks = None

    result = [labels, masks]

    return result


def main(img_url):
    img = cv2.imread(img_url)
    result = predict_and_output(img)

    if img_url[-1] == '/':
        filename = img_url.split('/')[-2]
    else:
        filename = img_url.split('/')[-1]

    output = show_predict_result.draw_polygon(img, result)
    output_path = os.path.join(str(config_dict['path']['output_path']), filename)

    cv2.imwrite(output_path, output)

    return output_path


if __name__ == '__main__':
    path = '/mnt/nfs_share/Dataset_results/soil_cover_0608/test'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # C01494388

    for f in onlyfiles:
        if 'C01494388' in f:
            continue

        img_path = os.path.join(path, f)
        result_path = main(img_path)
        print(result_path)

    # result_path = main('/mnt/nfs_share/Dataset_results/soil_cover_0608/test/C01494388_8_3.png')
