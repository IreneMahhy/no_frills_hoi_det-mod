import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers
from exp.hoi_classifier.models.verb_given_object_appearance import \
    VerbGivenObjectAppearanceConstants, VerbGivenObjectAppearance
from exp.hoi_classifier.models.verb_given_human_appearance import \
    VerbGivenHumanAppearanceConstants, VerbGivenHumanAppearance    
from exp.hoi_classifier.models.verb_given_boxes_and_object_label import \
    VerbGivenBoxesAndObjectLabelConstants, VerbGivenBoxesAndObjectLabel
from exp.hoi_classifier.models.verb_given_human_pose import \
    VerbGivenHumanPoseConstants, VerbGivenHumanPose
from exp.hoi_classifier.models.scatter_verbs_to_hois import \
    ScatterVerbsToHoisConstants, ScatterVerbsToHois


class HoiClassifierConstants(io.JsonSerializableClass):
    FACTOR_NAME_TO_MODULE_CONSTANTS = {
        'verb_given_object_app': VerbGivenObjectAppearanceConstants(),
        'verb_given_human_app': VerbGivenHumanAppearanceConstants(),
        'verb_given_boxes_and_object_label': VerbGivenBoxesAndObjectLabelConstants(),
        'verb_given_human_pose': VerbGivenHumanPoseConstants()
    }

    def __init__(self, data_sign='hico'):
        super(HoiClassifierConstants, self).__init__()
        self.data_sign = data_sign
        self.verb_given_appearance = True
        self.verb_given_human_appearance = True
        self.verb_given_object_appearance = True
        self.verb_given_boxes_and_object_label = True
        self.verb_given_human_pose = True
        self.rcnn_det_prob = True
        self.use_prob_mask = True
        self.use_object_label = True
        self.use_log_feat = True
        self.scatter_verbs_to_hois = self.get_scatter_con_func(self.data_sign)

    def get_scatter_con_func(self, data_sign):
        if data_sign == 'hico':
            return ScatterVerbsToHoisConstants()
        else:
            return None

    @property
    def selected_factor_constants(self):
        factor_constants = {}
        for factor_name in self.selected_factor_names:
            const = self.FACTOR_NAME_TO_MODULE_CONSTANTS[factor_name]
            factor_constants[factor_name] = const
        return factor_constants

    @property
    def selected_factor_names(self): 
        factor_names = []
        if self.verb_given_appearance:
            factor_names.append('verb_given_object_app')
            factor_names.append('verb_given_human_app')
        elif self.verb_given_human_appearance:
            factor_names.append('verb_given_human_app')
        elif self.verb_given_object_appearance:
            factor_names.append('verb_given_object_app')

        if self.verb_given_boxes_and_object_label:
            factor_names.append('verb_given_boxes_and_object_label')
        
        if self.verb_given_human_pose:
            factor_names.append('verb_given_human_pose')
        
        return factor_names


class HoiClassifier(nn.Module, io.WritableToFile):
    FACTOR_NAME_TO_MODULE = {
        'verb_given_object_app': VerbGivenObjectAppearance,
        'verb_given_human_app': VerbGivenHumanAppearance,
        'verb_given_boxes_and_object_label': VerbGivenBoxesAndObjectLabel,
        'verb_given_human_pose': VerbGivenHumanPose
    }

    def __init__(self, const, data_sign='hico'):
        super(HoiClassifier, self).__init__()
        self.const = copy.deepcopy(const)
        self.data_sign = data_sign
        self.sigmoid = pytorch_layers.get_activation('Sigmoid')
        self.scatter_verbs_to_hois = self.get_scatter_func(self.data_sign)
        for name, const in self.const.selected_factor_constants.items():
            self.create_factor(name, const)

    def get_scatter_func(self, data_sign):
        if data_sign == 'hico':
            return ScatterVerbsToHois(self.const.scatter_verbs_to_hois)
        else:
            return None

    def create_factor(self, factor_name, factor_const):
        if factor_name in ['verb_given_boxes_and_object_label',\
            'verb_given_human_pose']:
            factor_const.use_object_label = self.const.use_object_label
        if factor_name in ['verb_given_boxes_and_object_label']:
            factor_const.use_log_feat = self.const.use_log_feat
        factor = self.FACTOR_NAME_TO_MODULE[factor_name](factor_const)
        setattr(self, factor_name, factor)

    def forward(self, feats):
        factor_scores = {}
        any_verb_factor = False
        verb_factor_scores = 0
        verb_human_action_scores = 0  # sa`h，无object对应分数
        # Interaction Term中将human/object app, boxes, pose分数全部相加后通过sigmoid
        for factor_name in self.const.selected_factor_names:
            module = getattr(self, factor_name)
            factor_scores[factor_name] = module(feats)
            if 'verb_given' in factor_name:
                any_verb_factor = True
                verb_factor_scores += factor_scores[factor_name]
            if 'object' not in factor_name:
                verb_human_action_scores += factor_scores[factor_name]

        if any_verb_factor:
            verb_prob = self.sigmoid(verb_factor_scores)
            verb_human_action_prob = self.sigmoid(verb_human_action_scores)
            # 如果是hico数据集，将所有verb类别分数映射到所有hoi类别分数中
            if self.data_sign == 'hico':
                verb_prob_vec = self.scatter_verbs_to_hois(verb_prob)
            elif self.data_sign == 'vcoco':
                verb_prob_vec = verb_prob
                # verb_human_prob_vec = verb_human_action_prob
        else:
            verb_prob_vec = 0*feats['human_prob_vec'] + 1

        if self.const.rcnn_det_prob:
            human_prob_vec = feats['human_prob_vec']
            object_prob_vec = feats['object_prob_vec']
        else:
            human_prob_vec = 0*feats['human_prob_vec'] + 1
            object_prob_vec = 0*feats['object_prob_vec'] + 1

        prob_vec = {  # 分别为ho pair对应的human object verb score，no role则object prob全为1
            'human': human_prob_vec,
            'object': object_prob_vec,
            'verb': verb_prob_vec,
        }

        # 根据object_prob_vec，没有role的动作不会与object score相乘
        prob_vec['hoi'] = \
            prob_vec['human'] * \
            prob_vec['object'] * \
            prob_vec['verb']
        if self.const.use_prob_mask:  # 滤掉非候选的hoi
            prob_vec['hoi'] = prob_vec['hoi'] * feats['prob_mask']

        prob_vec['test_hoi'] = \
            prob_vec['human'] * \
            prob_vec['object'] * \
            prob_vec['verb']
        '''   
        if self.data_sign == 'vcoco':
            prob_vec['human_action'] = \
                prob_vec['human'] * \
                verb_human_prob_vec
        '''
        return prob_vec, factor_scores
