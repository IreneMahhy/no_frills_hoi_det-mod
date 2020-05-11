import os
import random

import utils.io as io
from data.vcoco.vcoco_constants import VcocoConstants


def split(global_ids, subsets):
    # val_frac is num_val / num_train_val

    split_ids = {
        'train': [],
        'val': [],
        'train_val': [],
        'test': []
    }

    for i, global_id in enumerate(global_ids):
        if subsets[i] == 'train':
            split_ids['train'].append(global_id)
            split_ids['train_val'].append(global_id)
        elif subsets[i] == 'val':
            split_ids['val'].append(global_id)
            split_ids['train_val'].append(global_id)
        else:
            split_ids['test'].append(global_id)

    return split_ids


def main():
    data_const = VcocoConstants()

    vcoco_list = io.load_json_object(data_const.anno_list_json)
    global_ids = [anno['global_id'] for anno in vcoco_list]
    subsets = [anno['subset'] for anno in vcoco_list]
    
    # Create and save splits
    split_ids = split(global_ids, subsets)

    split_ids_json = os.path.join(
        data_const.proc_dir,
        'split_ids.json')
    io.dump_json_object(split_ids, split_ids_json)

    # Create and save split stats
    split_stats = {}
    for subset_name, subset_ids in split_ids.items():
        split_stats[subset_name] = len(subset_ids)
        print(f'{subset_name}: {len(subset_ids)}')

    split_stats_json = os.path.join(
        data_const.proc_dir,
        'split_ids_stats.json')
    io.dump_json_object(split_stats, split_stats_json)


if __name__ == '__main__':
    main()
