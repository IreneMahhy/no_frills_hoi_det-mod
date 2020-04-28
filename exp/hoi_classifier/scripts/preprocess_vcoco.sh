SUBSETS=( 'train' 'val' 'test' )
for subset in "${SUBSETS[@]}"
do
    echo "Generate and label candidates for ${subset} ... "
    python -m exp.hoi_classifier.run \
        --exp exp_gen_and_label_hoi_cand_vcoco \
        --subset $subset \
        --gen_hoi_cand \
        --label_hoi_cand

    echo "Cache box features for ${subset} ... "
    python -m exp.hoi_classifier.run \
        --exp exp_cache_box_feats_vcoco \
        --subset $subset

    echo "Assign pose to human candidates for ${subset} ... "
    python -m exp.hoi_classifier.run \
        --exp exp_assign_pose_to_human_cand_vcoco \
        --subset $subset

    echo "Cache pose features for ${subset} ... "
    python -m exp.hoi_classifier.run \
        --exp exp_cache_pose_feats_vcoco \
        --subset $subset

done

