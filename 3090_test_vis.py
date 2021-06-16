import torch
import numpy as np
import pickle
import json
import detectron2
from detectron2.utils.logger import setup_logger
import matplotlib.pyplot as plt
import pandas as pd

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


def Train():
    register_coco_instances("custom", {},
                            "/mnt/xunjie_3090/soil_cover/train/COCO_segmentation.json",
                            "/mnt/xunjie_3090/soil_cover/train/train")
    custom_metadata = MetadataCatalog.get("custom")
    dataset_dicts = DatasetCatalog.get("custom")
    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=custom_metadata, scale=1)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('Sample', vis.get_image()[:, :, ::-1])
    #     cv2.waitKey()
    # cv2.destroyAllWindows()

    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    )
    cfg.DATASETS.TRAIN = ("custom",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = 'model_final_maskrcnn.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = (
        8000
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


def Predict():
    test_folder = '/mnt/xunjie_3090/soil_cover/test/test/'
    validate_folder = '/mnt/xunjie_3090/soil_cover/results/'

    register_coco_instances("test", {},
                            "/mnt/xunjie_3090/soil_cover/train/COCO_segmentation.json",
                            "/mnt/xunjie_3090/soil_cover/train/train")
    custom_metadata = MetadataCatalog.get("test")
    DatasetCatalog.get("test")

    image_names = []

    for root, dirs, files in os.walk(test_folder):
        for f in files:
            image_names.append(os.path.join(test_folder, f))

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TEST = ("test",)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    predictor = DefaultPredictor(cfg)

    for name in image_names:
        img = cv2.imread(name)
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1],
                       metadata=custom_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
                       )

        instance = outputs["instances"].to("cpu")

        scores = instance.scores if instance.has("scores") else None
        classes = instance.pred_classes if instance.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, custom_metadata.get("thing_classes", None))
        keypoints = instance.pred_keypoints if instance.has("pred_keypoints") else None
        output_info = VisImage(img[:, :, ::-1], scale=0.5)

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


if __name__ == '__main__':

    # print(dataframe[dataframe['iteration'] >= 4000])

    # for x in json_list:
    #     try:
    #
    #         plt.plot(x['iteration'], x['total_loss'])
    #         print(x['iteration'], x['total_loss'])
    #     except KeyError:
    #         print(x)

    # plt.plot(
    #     [x['iteration'] for x in json_list if 'iteration' in x],
    #     [x['total_loss'] for x in json_list if 'total_loss' in x])
    # plt.plot(
    #     [x['iteration'] for x in json_list if 'validation_loss' in x],
    #     [x['validation_loss'] for x in json_list if 'validation_loss' in x])
    # plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    # plt.show()


    # change_weights()
    # Train()
    Predict()
    # show_total_loss()
