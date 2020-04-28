import numpy as np
from tqdm import tqdm
import scipy.io as scio
import copy
from pycocotools.coco import COCO
from data.vcoco.vcoco_json import VCOCOeval

import utils.io as io
from data.vcoco.vcoco_constants import VcocoConstants
from data.vcoco_cfg import cfg


class Convert2Json:
    def __init__(self, const):
        self.const = const
        self.VCOCO_train = VCOCOeval(const.anno_vcoco_train, const.anno_list_trainval)
        self.VCOCO_val = VCOCOeval(const.anno_vcoco_val, const.anno_list_trainval)
        self.VCOCO_test = VCOCOeval(const.anno_vcoco_test, const.anno_list_test)

    def create_obj_list(self):
        object_list = list()
        COCO = self.VCOCO_train.COCO
        category_ids = COCO.getCatIds()
        categories = [c['name'] for c in COCO.loadCats(category_ids)]
        for i, ic in enumerate(category_ids):
            object_list_item = {
                'id': self.VCOCO_train.json_category_id_to_contiguous_id[ic],
                'name': categories[i]
            }
            object_list.append(object_list_item)

        return object_list

    def create_hoi_list(self):
        hoi_list = list()
        action_mask = np.array(cfg.VCOCO_ACTION_MASK).T
        has_role = np.where(action_mask == 1)
        hoi_idx = 0
        for i in range(has_role.shape[1]):
            hoi_idx = i + 1
            hoi_id = str(hoi_idx).zfill(3)  # 从1开始排序
            hoi_list_item = {
                'id': hoi_id,
                'object': True,
                'verb': self.VCOCO_train.actions[has_role[0][i]],  # 名字
            }
            hoi_list.append(hoi_list_item)

        for i, a in enumerate(action_mask):
            if np.all(a == 0):  # 该类动作没有role
                hoi_idx += 1
                hoi_id = str(hoi_idx).zfill(3)
                hoi_list_item = {
                    'id': hoi_id,
                    'object': False,
                    'verb': self.VCOCO_train.actions[i],
                }
                hoi_list.append(hoi_list_item)

        return hoi_list

    def get_hoi_bboxes(self, subset, entry):
        hois = list()
        pos_hoi_ids = []
        neg_hoi_ids = []
        hoi_idx = 0
        action_mask = np.array(cfg.VCOCO_ACTION_MASK).T
        has_role = np.where(action_mask == 1)

        for i in range(has_role.shape[1]):   # 利用action_mask寻找每个role作为一个类
            hoi_idx = i + 1
            hoi_id = str(hoi_idx).zfill(3)  # 从1开始排序
            action_i = has_role[0][i]  # 该类对应的action序号和role序号
            role_i = has_role[1][i]
            has_label = np.where(entry['gt_role_id'][:, action_i, role_i] != -1)[0]  # 在该位置存在role
            if has_label.size > 0:
                pos_hoi_ids.append(hoi_id)  # 表明图片中存在该类
                object_id = entry['gt_role_id'][has_label, action_i, role_i]
                human_bbox_idx = np.where(entry['gt_classes'] == 1)[0]
                object_bbox_idx = np.where(entry['gt_classes'] != 1)[0]
                human_bboxes = entry['boxes'][human_bbox_idx]
                object_bboxes = entry['boxes'][object_bbox_idx]
                bbox_id_to_human_id = {  # 由entry['boxes'] id映射到human_bboxes中的id
                    y: x for x, y in enumerate(human_bbox_idx)}
                bbox_id_to_object_id = {  # 由entry['boxes'] id映射到human_bboxes中的id
                    y: x for x, y in enumerate(object_bbox_idx)}

                connections = []
                for j, idx in enumerate(has_label):
                    if entry['gt_classes'][idx] == 1 and entry['gt_classes'][object_id[j]] != 1:
                        x1 = bbox_id_to_human_id[idx]
                        x2 = bbox_id_to_object_id[object_id[j]]
                        connections.append((x1, x2))
                hoi = {
                    'id': hoi_id,
                    'human_bboxes': human_bboxes,
                    'object_bboxes': object_bboxes,
                    'connections': connections,
                    'invis': 0,  # ?
                }
                hois.append(hoi)
            else:
                neg_hoi_ids.append(hoi_id)

        for i, a in enumerate(action_mask):
            if np.all(a == 0):  # 该类动作没有role
                hoi_idx += 1
                hoi_id = str(hoi_idx).zfill(3)
                has_label = np.where(entry['gt_actions'][:, i] == 1)[0]
                if has_label.size > 0:
                    pos_hoi_ids.append(hoi_id)
                    human_bbox_idx = np.where(entry['gt_classes'] == 1)[0]
                    human_bboxes = entry['boxes'][human_bbox_idx]
                    bbox_id_to_human_id = {  # 由entry['boxes'] id映射到human_bboxes中的id
                        y: x for x, y in enumerate(human_bbox_idx)}

                    connections = []
                    for j in has_label:
                        if entry['gt_classes'][j] == 1:
                            x1 = bbox_id_to_human_id[j]
                            connections.append((x1, -1))
                    hoi = {
                        'id': hoi_id,
                        'human_bboxes': human_bboxes,
                        'object_bboxes': [],
                        'connections': connections,
                        'invis': 0,  # ?
                    }
                    hois.append(hoi)
                else:
                    neg_hoi_ids.append(hoi_id)
        return hois, pos_hoi_ids, neg_hoi_ids


    def create_anno_list(self):
        anno_list = []
        for subset in ['train', 'val', 'test']:
            print(f'Adding {subset} data to anno list ...')
            if subset == 'train':
                VCOCO = self.VCOCO_train
                image_dir_prefix = 'train2014'
            elif subset == 'val':
                VCOCO = self.VCOCO_val
                image_dir_prefix = 'train2014'
            else:
                VCOCO = self.VCOCO_test
                image_dir_prefix = 'val2014'

            vcocodb = VCOCO.vcocodb
            for entry in tqdm(vcocodb):
                image_jpg = entry['file_name']
                if image_jpg.endswith('.jpg'):
                    global_id = image_jpg[:-4]
                else:
                    assert False, 'Image extension is not .jpg'

                image_size = [int(v) for v in [entry['width'], entry['height'], 3]]
                hois, pos_hoi_ids, neg_hoi_ids = self.get_hoi_bboxes(subset, entry)
                anno = {
                    'global_id': global_id,
                    'image_path_postfix': f'{image_dir_prefix}/{image_jpg}',
                    'image_size': image_size,
                    'hois': hois,
                    'subset': subset,
                    'pos_hoi_ids': pos_hoi_ids,
                    'neg_hoi_ids': neg_hoi_ids
                }
                anno_list.append(anno)

        return anno_list

    def convert(self):
        print('Creating anno list for vcoco...')
        anno_list = self.create_anno_list()
        io.dump_json_object(anno_list, self.const.anno_list_json)

        print('Creating hoi list for vcoco...')
        hoi_list = self.create_hoi_list()
        io.dump_json_object(hoi_list, self.const.hoi_list_json)

        print('Creating object list for vcoco...')
        object_list = self.create_obj_list()
        io.dump_json_object(object_list, self.const.object_list_json)

        print('Creating verb list for vcoco...')
        verb_list = []
        for i, verb in enumerate(self.VCOCO_train.actions):
            verb_list_item = {
                'id': self.VCOCO_train.actions_to_id_map[verb],
                'name': verb
            }
            verb_list.append(verb_list_item)
        io.dump_json_object(verb_list, self.const.verb_list_json)


if __name__ == '__main__':
    vcoco_const = VcocoConstants()
    converter = Convert2Json(vcoco_const)
    converter.convert()
