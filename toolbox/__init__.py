from .metrics import averageMeter, runningScore
from .log import get_logger
import torch.nn as nn


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', "glassrgbt", "mirrorrgbd", 'glassrgbt_merged', 'trosd']

    if cfg['dataset'] == 'mirrorrgbd':
        from .datasets.mirrorrgbd import MirrorRGBD
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return MirrorRGBD(cfg, mode='train'), MirrorRGBD(cfg, mode='test')

    if cfg['dataset'] == 'trosd':
        from .datasets.trosd import TROSDRGBD
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return TROSDRGBD(cfg, mode='train'), TROSDRGBD(cfg, mode='test')




def get_model(cfg):

    if cfg['model_name'] == 'HAFNet':
        from toolbox.models.HAFNet import HAFNet
        return HAFNet()





