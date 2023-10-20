import argparse
import glob
import logging
import os
import os.path as osp
import pickle

import cv2
import numpy as np
import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.layers import nms
from detectron2.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.det_thresh  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml")
    predictor = DefaultPredictor(cfg)
    img_list = glob.glob(osp.join(args.img_dir, '**', '*.png'), recursive=True) + glob.glob(osp.join(args.img_dir, '**', '*.jpg'), recursive=True) + glob.glob(osp.join(args.img_dir, '**', '*.jpeg'), recursive=True)
    logger.info('Now processing dir %s, %d images found.', args.img_dir, len(img_list))
    bbox_data = {}
    image_reuse_cnt = {}
    empty_box = 0
    for ind, file in tqdm.tqdm(enumerate(img_list)):
        img = cv2.imread(file)
        img_uid = file.split('/')[-1]
        outputs = predictor(img)
        # select_index = nms(outputs['proposals'].proposal_boxes.tensor, outputs['proposals'].objectness_logits, args.nms)
        bbox = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        # bbox = outputs["proposals"].get('proposal_boxes')[select_index].tensor.cpu().numpy()[:args.topk]
        if img_uid in bbox_data:
            if img_uid in image_reuse_cnt:
                image_reuse_cnt[img_uid] += 1
            else:
                image_reuse_cnt[img_uid] = 1
            continue
        bbox_data[img_uid] = {}
        if bbox.shape[0] == 0:
            # insert two full boxes for RN
            bbox_data[img_uid]['boxes'] = np.array([
                    [0, 0, img.shape[1], img.shape[0]],
                    [0, 0, img.shape[1], img.shape[0]],
                ], dtype=np.float32)
            empty_box += 1
        else:
            bbox_data[img_uid]['boxes'] = bbox
        # viz_data = Instances(img.shape[:2], pred_boxes=Boxes(bbox))
        if args.viz:
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            viz_out = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
            # viz_out = v.draw_instance_predictions(viz_data.to("cpu")).get_image()[:, :, ::-1]
            cv2.imwrite(osp.join(args.save_dir, img_uid), viz_out)
        if not ind % 100:
            pickle.dump(bbox_data, open(osp.join(args.save_dir, 'bbox_data.pkl'), 'wb'))
    pickle.dump(bbox_data, open(osp.join(args.save_dir, 'bbox_data.pkl'), 'wb'))
    logger.info('The following images have been reused: %s', image_reuse_cnt)
    logger.info(f'No box for {empty_box} out of {len(img_list)} images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default=None, required=True)
    parser.add_argument('--save_dir', default='./bbox_data')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--nms', type=float, default=0.3)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--det_thresh', type=float, default=0.7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)