from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = 'NumberPlaceDataset'
_C.DATASET.ROOT = '../data/' + _C.DATASET.NAME
_C.DATASET.NUM_CLASSES = 10
_C.DATASET.BATCH_SIZE = 4
_C.DATASET.TOTAL_EPOCH = 10000

_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.LOG_DIR = '../logs'
_C.MODEL.PRETRAINED = ''
_C.MODEL.OPTIMIZER = ''
_C.MODEL.CRITERION = ''
_C.MODEL.INPUT_SIZE = (28, 28)

_C.SCHEDULER = CN(new_allowed=True)
_C.SCHEDULER.NAME = ''


def get_defaults():
    return _C.clone()


def load_config(config_path):
    cfg = get_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg
