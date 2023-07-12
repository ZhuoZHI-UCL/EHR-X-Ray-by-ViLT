#这个文件中我们直接引用 Extract_data_1的函数来获取dataloader
#dataloader的参数可以在config中调节
import os, sys
# 将当前文件的路径添加到sys.path

import functools
from pytorch_lightning import LightningDataModule
sys.path.append('../..')
from Test.Extract_data_1 import get_dataloader



class MTDataModule(LightningDataModule):
    def __init__(self,config,dist=False):#dist：是否使用分布式训练
        super().__init__()
        self.train_dl, self.val_dl, self.test_dl =  get_dataloader(config)
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl


       