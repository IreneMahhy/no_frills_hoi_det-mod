import os
import h5py
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tensorboard_logger import configure, log_value

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
from exp.hoi_classifier.models.hoi_classifier_model import HoiClassifier
from exp.hoi_classifier.data.features_dataset import Features
from data.vcoco_cfg import cfg


def eval_model(model, dataset, exp_const, data_sign):
    print('Creating hdf5 file for predicted hoi dets ...')
    if data_sign == 'hico':
        pred_hoi_dets_hdf5 = os.path.join(
            exp_const.exp_dir,
            f'pred_hoi_dets_{dataset.const.subset}_{model.const.model_num}.hdf5')
        pred_hois = h5py.File(pred_hoi_dets_hdf5, 'w')
    else:
        pred_hoi_dets_hdf5_vcoco = os.path.join(
            exp_const.exp_dir,
            f'pred_hoi_dets_{dataset.const.subset}_{model.const.model_num}.hdf5')
        dets = h5py.File(pred_hoi_dets_hdf5_vcoco, 'w')

    model.hoi_classifier.eval()
    sampler = SequentialSampler(dataset)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]

        with torch.no_grad():
            feats = {
                'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
                'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat'])),
                'box': Variable(torch.cuda.FloatTensor(data['box_feat'])),
                'absolute_pose': Variable(torch.cuda.FloatTensor(data['absolute_pose'])),
                'relative_pose': Variable(torch.cuda.FloatTensor(data['relative_pose'])),
                'human_prob_vec': Variable(torch.cuda.FloatTensor(data['human_prob_vec'])),
                'object_prob_vec': Variable(torch.cuda.FloatTensor(data['object_prob_vec'])),
                'object_one_hot': Variable(torch.cuda.FloatTensor(data['object_one_hot'])),
                'prob_mask': Variable(torch.cuda.FloatTensor(data['prob_mask']))
            }

        prob_vec, factor_scores = model.hoi_classifier(feats)
        
        hoi_prob = prob_vec['test_hoi']  # 没有加prob_mask
        hoi_prob = hoi_prob.data.cpu().numpy()

        if data_sign == 'vcoco':
            num_action_classes = cfg.VCOCO_NUM_ACTION_CLASSES
            num_target_object_types = cfg.VCOCO_NUM_TARGET_OBJECT_TYPES
            action_mask = np.array(cfg.VCOCO_ACTION_MASK).T

            unique_pair_index = data['unique_index']
            no_role_index = data['no_role_index']
            unique_prob = hoi_prob[unique_pair_index]
            no_role_prob = hoi_prob[no_role_index]  # no role pair with human only, dim0=human_num
            human_boxes, object_boxes = data['human_boxes'], data['object_boxes']
            # dim0为human_num*object_num，得到所有区分role的动作的预测分数
            hoi_prob_tmp = np.zeros(
                (unique_prob.shape[0], num_action_classes, num_target_object_types),
                dtype=hoi_prob.dtype)
            hoi_prob_tmp[:, np.where(action_mask > 0)[0], np.where(action_mask > 0)[1]] = \
                unique_prob[:, :-5]
            hoi_prob = hoi_prob_tmp.reshape(len(human_boxes), len(object_boxes), num_action_classes, num_target_object_types)
            choosed_object_inds = np.argmax(hoi_prob, axis=1)
            hoi_prob = np.max(hoi_prob, axis=1)
            choosed_objects = object_boxes[choosed_object_inds]

            # agents: box coordinates + 26 action score.
            agents = np.hstack((human_boxes, no_role_prob))
            roles = np.concatenate((choosed_objects, hoi_prob[..., np.newaxis]), axis=-1)
            roles = np.stack(
                [roles[:, :, i, :].reshape(-1, num_action_classes * 5) for i in range(num_target_object_types)],
                axis=-1)

            global_id = data['global_id']
            dets.create_group(global_id)
            dets[global_id].create_dataset('agents', data=agents)
            dets[global_id].create_dataset('roles', data=roles)

        else:
            num_cand = hoi_prob.shape[0]
            scores = hoi_prob[np.arange(num_cand), np.array(data['hoi_idx'])]
            human_obj_boxes_scores = np.concatenate((
                data['human_box'],
                data['object_box'],
                np.expand_dims(scores, 1)), 1)

            global_id = data['global_id']
            pred_hois.create_group(global_id)
            pred_hois[global_id].create_dataset(
                'human_obj_boxes_scores',
                data=human_obj_boxes_scores)
            pred_hois[global_id].create_dataset(
                'start_end_ids',
                data=data['start_end_ids_'])

    if data_sign == 'hico':
        pred_hois.close()
    else:
        dets.close()


def main(exp_const, data_const, model_const, data_sign):
    print('Loading model ...')
    model = Model()
    model.const = model_const
    model.hoi_classifier = HoiClassifier(model.const.hoi_classifier, data_sign).cuda()
    if model.const.model_num == -1:
        print('No pretrained model will be loaded since model_num is set to -1')
    else:
        model.hoi_classifier.load_state_dict(
            torch.load(model.const.hoi_classifier.model_pth))

    print('Creating data loader ...')
    dataset = Features(data_const)


    eval_model(model, dataset, exp_const, data_sign)
