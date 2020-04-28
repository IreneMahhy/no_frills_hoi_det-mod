from utils.collections import AttrDict

__C = AttrDict()

cfg = __C

__C.VCOCO_NUM_ACTION_CLASSES = 26

__C.VCOCO_NUM_TARGET_OBJECT_TYPES = 2

__C.VCOCO_ACTION_MASK = [[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

__C.VCOCO_NO_ROLE_ACTION_NUM = 5

__C.VCOCO_ACTION_NUM_WITH_ROLE = 29
