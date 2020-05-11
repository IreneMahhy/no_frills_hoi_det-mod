import os
import numpy as np
from tqdm import tqdm
import threading
import h5py

import utils.io as io
from utils.constants import save_constants
from data.coco_classes import COCO_CLASSES


class HoiCandidatesGenerator():
    def __init__(self, data_const, data_sign='hico'):
        self.data_const = data_const
        self.data_sign = data_sign
        self.hoi_classes = self.get_hoi_classes()
        
    def get_hoi_classes(self):
        hoi_list = io.load_json_object(self.data_const.hoi_list_json)
        hoi_classes = {hoi['id']: hoi for hoi in hoi_list}
        return hoi_classes

    def predict(self, selected_dets):
        pred_hoi_dets = []
        start_end_ids = np.zeros([len(self.hoi_classes), 2], dtype=np.int32)
        start_id = 0
        for hoi_id, hoi_info in self.hoi_classes.items():
            dets = self.predict_hoi(selected_dets, hoi_info)
            pred_hoi_dets.append(dets)
            hoi_idx = int(hoi_id)-1
            start_end_ids[hoi_idx, :] = [start_id, start_id+dets.shape[0]]
            start_id += dets.shape[0]
        pred_hoi_dets = np.concatenate(pred_hoi_dets)
        return pred_hoi_dets, start_end_ids

    def predict_hoi(self, selected_dets, hoi_info):
        human_boxes = selected_dets['boxes']['person']
        human_scores = selected_dets['scores']['person']
        human_rpn_ids = selected_dets['rpn_ids']['person']

        if self.data_sign == 'hico':
            # 用空格替代object name中的-符号
            hoi_object = ' '.join(hoi_info['object'].split('_'))
            object_boxes = selected_dets['boxes'][hoi_object]
            object_scores = selected_dets['scores'][hoi_object]
            object_rpn_ids = selected_dets['rpn_ids'][hoi_object]
            num_hoi_dets = human_boxes.shape[0] * object_boxes.shape[0]
            hoi_dets = np.zeros([num_hoi_dets, 13])
            hoi_idx = int(hoi_info['id']) - 1
            hoi_dets[:, -1] = hoi_idx
            count = 0
            for i in range(human_boxes.shape[0]):
                for j in range(object_boxes.shape[0]):
                    hoi_dets[count, :4] = human_boxes[i]
                    hoi_dets[count, 4:8] = object_boxes[j]
                    hoi_dets[count, 8:12] = [human_scores[i], object_scores[j], \
                                             human_rpn_ids[i], object_rpn_ids[j]]
                    count += 1
        else:
            object_box_list = []
            object_score_list = []
            object_id_list = []
            for cls_name in COCO_CLASSES:
                if cls_name == 'person':
                    continue
                object_box_list.append(selected_dets['boxes'][cls_name])
                object_score_list.append(selected_dets['scores'][cls_name])
                object_id_list.append(selected_dets['rpn_ids'][cls_name])
            object_boxes = np.concatenate(object_box_list)
            object_scores = np.concatenate(object_score_list)
            object_rpn_ids = np.concatenate(object_id_list)

            hoi_idx = int(hoi_info['id']) - 1
            if not hoi_info['object']:  # 没有role的动作，候选对即所有的human box
                num_hoi_dets = human_boxes.shape[0]
                hoi_dets = np.zeros([num_hoi_dets, 13])
                hoi_dets[:, -1] = hoi_idx
                for i in range(human_boxes.shape[0]):
                    hoi_dets[i, :4] = human_boxes[i]
                    hoi_dets[i, 4:8] = [0, 0, 0, 0]
                    hoi_dets[i, 8:12] = [human_scores[i], 0, human_rpn_ids[i], -1]
            else:
                num_hoi_dets = human_boxes.shape[0] * object_boxes.shape[0]
                hoi_dets = np.zeros([num_hoi_dets, 13])
                hoi_dets[:, -1] = hoi_idx
                count = 0
                for i in range(human_boxes.shape[0]):
                    for j in range(object_boxes.shape[0]):
                        hoi_dets[count, :4] = human_boxes[i]
                        hoi_dets[count, 4:8] = object_boxes[j]
                        hoi_dets[count, 8:12] = [human_scores[i], object_scores[j], \
                                                 human_rpn_ids[i], object_rpn_ids[j]]
                        count += 1
        return hoi_dets


def generate(exp_const, data_const, data_sign):
    print(f'Creating exp_dir: {exp_const.exp_dir}')
    io.mkdir_if_not_exists(exp_const.exp_dir)

    save_constants({'exp': exp_const, 'data': data_const}, exp_const.exp_dir)

    print(f'Reading split_ids.json ...')
    split_ids = io.load_json_object(data_const.split_ids_json)

    print('Creating an object-detector-only HOI detector ...')
    hoi_cand_gen = HoiCandidatesGenerator(data_const, data_sign)

    print(f'Creating a hoi_candidates_{exp_const.subset}.hdf5 file ...')
    hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir, f'hoi_candidates_{exp_const.subset}.hdf5')
    f = h5py.File(hoi_cand_hdf5, 'w')

    # 从Faster RCNN的所有预测结果中选择的高分预测
    print('Reading selected dets from hdf5 file ...')
    all_selected_dets = h5py.File(data_const.selected_dets_hdf5, 'r')

    for global_id in tqdm(split_ids[exp_const.subset]):
        selected_dets = {
            'boxes': {},
            'scores': {},
            'rpn_ids': {}
        }
        start_end_ids = all_selected_dets[global_id]['start_end_ids'][()]
        boxes_scores_rpn_ids = \
            all_selected_dets[global_id]['boxes_scores_rpn_ids'][()]

        for cls_ind, cls_name in enumerate(COCO_CLASSES):
            start_id, end_id = start_end_ids[cls_ind]
            boxes = boxes_scores_rpn_ids[start_id:end_id, :4]
            scores = boxes_scores_rpn_ids[start_id:end_id, 4]
            rpn_ids = boxes_scores_rpn_ids[start_id:end_id, 5]
            selected_dets['boxes'][cls_name] = boxes
            selected_dets['scores'][cls_name] = scores
            selected_dets['rpn_ids'][cls_name] = rpn_ids

        pred_dets, start_end_ids = hoi_cand_gen.predict(selected_dets)
        f.create_group(global_id)
        f[global_id].create_dataset(
            'boxes_scores_rpn_ids_hoi_idx', data=pred_dets)
        f[global_id].create_dataset('start_end_ids', data=start_end_ids)

    f.close()
