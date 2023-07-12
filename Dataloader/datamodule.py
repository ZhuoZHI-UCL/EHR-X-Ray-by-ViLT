from pytorch_lightning import LightningDataModule
from  Dataloader.Extract_data_1  import get_dataloader

class datamodule(LightningDataModule):
    def __init__(self,config,dist=False):#dist：是否使用分布式训练
        super().__init__()
        self.config = config
        self.train_dl, self.val_dl, self.test_dl =  get_dataloader(config)

    def train_dataloader(self):
        return self.train_dl


    def val_dataloader(self):
    
        if self.config.data_pairs=='radiology':
            return self.val_dl
        else:
            return self.test_dl

    def test_dataloader(self):
        return self.test_dl