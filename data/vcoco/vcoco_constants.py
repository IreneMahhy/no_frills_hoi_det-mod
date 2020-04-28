import os


class VcocoConstants:
    def __init__(
            self,
            clean_dir=os.path.join(os.getcwd(), 'data_symlinks/vcoco_clean'),
            proc_dir=os.path.join(os.getcwd(), 'data_symlinks/vcoco_processed')):
        self.clean_dir = clean_dir
        self.proc_dir = proc_dir

        # Clean constants
        self.anno_list_trainval = os.path.join(self.clean_dir, 'vcoco', 'annotations',
                                                'instances_vcoco_trainval_2014.json')
        self.anno_list_test = os.path.join(self.clean_dir, 'vcoco', 'annotations',
                                            'instances_vcoco_test_2014.json')
        self.anno_vcoco_train = os.path.join(self.clean_dir, 'vcoco', 'vcoco',
                                                'vcoco_train.json')
        self.anno_vcoco_val = os.path.join(self.clean_dir, 'vcoco', 'vcoco',
                                                'vcoco_val.json')
        self.anno_vcoco_test = os.path.join(self.clean_dir, 'vcoco', 'vcoco',
                                                'vcoco_test.json')
        self.images_dir = os.path.join(self.clean_dir, 'images')

        # Processed constants
        self.anno_list_json = os.path.join(self.proc_dir, 'anno_list.json')
        self.hoi_list_json = os.path.join(self.proc_dir, 'hoi_list.json')
        self.object_list_json = os.path.join(self.proc_dir, 'object_list.json')
        self.verb_list_json = os.path.join(self.proc_dir, 'verb_list.json')

        # Need to run split_ids.py
        self.split_ids_json = os.path.join(self.proc_dir, 'split_ids.json')

        # Need to run hoi_cls_count.py
        self.hoi_cls_count_json = os.path.join(self.proc_dir, 'hoi_cls_count.json')
        self.bin_to_hoi_ids_json = os.path.join(self.proc_dir, 'bin_to_hoi_ids.json')

        self.faster_rcnn_boxes = os.path.join(self.proc_dir, 'faster_rcnn_boxes')
