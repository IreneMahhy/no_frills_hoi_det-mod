#!/bin/bash
SUBSET="test"
VCOCO_EXP_DIR="${PWD}/data_symlinks/vcoco_exp/hoi_classifier"
EXP_NAME="factors_rcnn_det_prob_appearance_boxes_and_object_label_human_pose"
echo $EXP_NAME
MODEL_NUM="30000"
PRED_HOI_DETS_HDF5="${VCOCO_EXP_DIR}/${EXP_NAME}/pred_hoi_dets_${SUBSET}_${MODEL_NUM}.hdf5"
OUT_DIR="${VCOCO_EXP_DIR}/${EXP_NAME}/mAP_eval/${SUBSET}_${MODEL_NUM}"
PROC_DIR="${PWD}/data_symlinks/vcoco_processed"

python -m exp.hico_eval.compute_map_vcoco \
    --pred_hoi_dets_hdf5 $PRED_HOI_DETS_HDF5 \
    --out_dir $OUT_DIR \
    --proc_dir $PROC_DIR \
    --subset $SUBSET

python -m exp.hico_eval.sample_complexity_analysis \
    --out_dir $OUT_DIR
