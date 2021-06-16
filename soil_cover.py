import torch
import numpy as np
import pickle
import json
import detectron2
from detectron2.utils.logger import setup_logger
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import configparser

setup_logger()

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
from detectron2.utils.visualizer import ColorMode
import uuid
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import _create_text_labels, VisImage, GenericMask

from PIL import Image

import pprint


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def info():
    config_dict = dict()
    config_dict['path'] = {
        'dataset_folder': '/mnt/nfs_share/Xunjie/Soil_cover/soil_cover_0412',
        'test_folder': '/home/xunjie/Soil Cover/test/'
    }
    config_dict['parameters'] = {
        'NUM_WORKERS': 16,
        'IMS_PER_BATCH': 4,
        'BASE_LR': 0.01,
        'IMAGE_MIN_DIM': 1024,
        'IMAGE_MAX_DIM': 1920,
        'SOLVER_MAX_ITER': 8000,
        'BATCH_SIZE_PER_IMAGE': 512,
        'NUM_CLASSES': 4
    }

    manager = ConfigManager()
    manager.dump(config_dict, './soil_cover.ini')



def Train():
    manager = ConfigManager()
    config_dict = manager.load('/home/fei/Xunjie/detectron2/soil_cover.ini')
    dataset_folder = str(config_dict['path']['dataset_folder'])
    train_json = os.path.join(dataset_folder, 'COCO_segmentation.json')

    if len(dataset_folder.split('/')[-1]) < 2:
        folder_name = dataset_folder.split('/')[-2]
    else:
        folder_name = dataset_folder.split('/')[-1]

    train_images = os.path.join(dataset_folder, folder_name)
    register_coco_instances("custom", {}, train_json, train_images)

    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    )
    cfg.DATASETS.TRAIN = ("custom",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = int(config_dict['parameters']['num_workers'])
    cfg.MODEL.WEIGHTS = 'model_final_maskrcnn.pkl'
    cfg.SOLVER.IMS_PER_BATCH = int(config_dict['parameters']['ims_per_batch'])
    cfg.SOLVER.BASE_LR = float(config_dict['parameters']['base_lr'])
    cfg.IMAGE_MIN_DIM = int(config_dict['parameters']['image_min_dim'])
    cfg.IMAGE_MAX_DIM = int(config_dict['parameters']['image_max_dim'])
    cfg.SOLVER.MAX_ITER = (
        int(config_dict['parameters']['solver_max_iter'])
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        int(config_dict['parameters']['batch_size_per_image'])
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(config_dict['parameters']['num_classes'])

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


def Predict():
    dataset_folder = '/mnt/xunjie_3090/Dataset_results/soil_cover_0520'
    train_json = os.path.join(dataset_folder, 'COCO_segmentation.json')
    train_images = os.path.join(dataset_folder, dataset_folder.split(   '/')[-1])

    test_folder = '/mnt/xunjie_3090/Dataset_results/Soil_cover/soil_cover_0520/soil_cover_0520'
    validate_folder = '/mnt/xunjie_3090/Xunjie/Soil_cover/soil_cover_0520/test'

    register_coco_instances("test", {},
                            train_json,
                            train_images)
    custom_metadata = MetadataCatalog.get("test")
    DatasetCatalog.get("test")

    image_names = []

    for root, dirs, files in os.walk(test_folder):
        for f in files:
            image_names.append(os.path.join(test_folder, f))

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TEST = ("test",)
    cfg.MODEL.WEIGHTS = '/mnt/xunjie_3090/Xunjie/Soil_cover/soil_cover_0520/weights/model_40000.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512
    )
    cfg.INPUT.MIN_SIZE_TEST = 1920
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    predictor = DefaultPredictor(cfg)

    for name in image_names:
        img = cv2.imread(name)
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1],
                       metadata=custom_metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
                       )

        instance = outputs["instances"].to("cpu")

        scores = instance.scores if instance.has("scores") else None
        classes = instance.pred_classes if instance.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, custom_metadata.get("thing_classes", None))
        keypoints = instance.pred_keypoints if instance.has("pred_keypoints") else None
        output_info = VisImage(img[:, :, ::-1], scale=1)

        if instance.has("pred_masks"):
            masks = np.asarray(instance.pred_masks)
            masks = [GenericMask(x, output_info.height, output_info.width) for x in masks]
        else:
            masks = None

        # print(len(masks[0].polygons))
        # print(keypoints)
        print("Len of labels: ", len(labels))
        # Labels:  ['steel plate 100%', 'green network 99%', 'soil 99%', 'concrete 98%']
        # Polygons in the same order
        # Something like this
        # Labels:  ['green network 100%', 'concrete 100%', 'soil 100%', 'green network 50%']
        # the len of masks == len of labels
        print("Labels: ", labels)
        print("classes: ", classes)
        for i in masks:
            print("Len, ", len(i.polygons))

        v = v.draw_instance_predictions(instance)
        result_image = v.get_image()[:, :, ::-1]
        filename = name.split('/')[-1]
        cv2.imwrite(os.path.join(validate_folder, filename), result_image)
        print("%s saved" % filename)


def change_weights():
    num_class = 4
    with open('checkpoints/model_final_a54504.pkl', 'rb') as f:
        obj = f.read()
    weights = pickle.loads(obj, encoding='latin1')

    weights['model']['roi_heads.box_predictor.cls_score.weight'] = np.zeros([num_class + 1, 1024], dtype='float32')
    weights['model']['roi_heads.box_predictor.cls_score.bias'] = np.zeros([num_class + 1], dtype='float32')

    weights['model']['roi_heads.box_predictor.bbox_pred.weight'] = np.zeros([num_class * 4, 1024], dtype='float32')
    weights['model']['roi_heads.box_predictor.bbox_pred.bias'] = np.zeros([num_class * 4], dtype='float32')

    weights['model']['roi_heads.mask_head.predictor.weight'] = np.zeros([num_class, 256, 1, 1], dtype='float32')
    weights['model']['roi_heads.mask_head.predictor.bias'] = np.zeros([num_class], dtype='float32')

    f = open('model_final_maskrcnn.pkl', 'wb')
    pickle.dump(weights, f)
    f.close()


def show_total_loss():
    metric_path = './output/metrics.json'
    json_list = []

    with open(metric_path, 'r') as f:
        for line in f:
            json_list.append(json.loads(line))

    for j in json_list:
        if len(j) != 21:
            json_list.remove(j)

    dataframe = pd.DataFrame(json_list)
    dataframe.plot(x='iteration', y='total_loss', kind='line')

    plt.show()


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


if __name__ == '__main__':
    Train()
    # shutil.copy(src='./output/model_final.pth', dst='/mnt/xunjie_3090/soil_cover_models/3090/output_0322/')
    # Predict()
    # show_total_loss()
    # info()
