import os
import yaml

class Config(object):
    def __init__(self):
        # config file
        with open(os.path.join('config.yml'), 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        self.batch_size = cfg['batch_size']
        self.height = cfg['height']
        self.width = cfg['width']
        self.epochs = cfg['epochs']
        self.patient = cfg['patient']
        self.opt = cfg['optimizer']
        self.lr = cfg['lr']
        self.quick_train = cfg['quick_train']
        self.augment = cfg['augment']