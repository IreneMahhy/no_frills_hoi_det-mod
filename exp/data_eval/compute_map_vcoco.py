import os
import argparse
import time
import h5py
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

import utils.io as io
from data.vcoco.vcoco_constants import VcocoConstants
from data.vcoco.vcoco_json import VCOCOeval

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pred_hoi_dets_hdf5',
    type=str,
    default=None,
    required=True,
    help='Path to predicted hoi detections hdf5 file')
parser.add_argument(
    '--out_dir',
    type=str,
    default=None,
    required=True,
    help='Output directory')
parser.add_argument(
    '--proc_dir',
    type=str,
    default=None,
    required=True,
    help='Path to processed data directory')
parser.add_argument(
    '--subset',
    type=str,
    default='test',
    choices=['train', 'test', 'val', 'train_val'],
    help='Subset of data to run the evaluation on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=12,
    help='Number of processes to parallelize across')


def main():
    args = parser.parse_args()

    print('Creating output dir ...')
    # io.mkdir_if_not_exists(args.out_dir, recursive=True)

    pred_dets = h5py.File(args.pred_hoi_dets_hdf5, 'r')

    vcoco_const = VcocoConstants()
    vcoco_test = VCOCOeval(vcoco_const.anno_vcoco_test, vcoco_const.anno_list_test)
    vcoco_eval = vcoco_test.do_eval(pred_dets)
'''
    mAP_json = os.path.join(
        args.out_dir,
        'mAP.json')

    print(f'APs have been saved to {args.out_dir}')
'''

if __name__ == '__main__':
    main()
