import pytorch_lightning as pl
from Model.tools import tools_ehr, tools_metric
import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def replace_outliers_with_zero(x):
    mask = (x < 100) | (x > -100)
    x[mask] = 0
    return x
def check_and_print_nan(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f'{tensor_name} contains NaN values.')

class ehr_only_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters() #这个函数可以直接保存所有的入口参数(config)到checkpoint中    
        self.max_steps = self.hparams.config.max_steps
        self.task=self.hparams.config.task
        self.ehr_model = tools_ehr.LSTM_Medfuse( )

        #定义一些属性，在每个step中更新，在每个epoch结束后计算
        self.outGT_train = torch.FloatTensor()
        self.outPRED_train = torch.FloatTensor()

        self.outGT_val = torch.FloatTensor()
        self.outPRED_val = torch.FloatTensor()

        self.outGT_test = torch.FloatTensor()
        self.outPRED_test = torch.FloatTensor()


    def forward(self, batch):

        #----------------------------------------------从batch中提取要用的数据----------------------------------------------#
        # ehr_norm=replace_outliers_with_zero(batch[0])
        # padding_size=self.hparams.config.ehr_max_len - ehr_norm.shape[1]
        # if padding_size > 0:
        #     # 在seq_len维度进行padding
        #     ehr_norm= torch.from_numpy(ehr_norm).float()
        #     ehr_data = F.pad(ehr_norm, (0, 0, 0, padding_size))

        # else:
        #     ehr_data=torch.from_numpy(ehr_norm[:, :self.hparams.config.ehr_max_len, :]).float()
        
        # seq_length=torch.tensor(batch[4])

        ehr_data=torch.from_numpy(batch[0]).float()
        seq_length=torch.tensor(batch[4])
        #-----------------------------------------------------数据的处理---------------------------------------------------#
        pre_ehr,ehr_embeds =  self.ehr_model(ehr_data,seq_length)


        #-----------------------------------------------------返回结果-----------------------------------------------------#
        return pre_ehr,ehr_embeds

    def training_step(self, batch, batch_idx):#定义训练过程每一步
        #读取数据并预处理
        y_hat,_ = self(batch)
        y_hat = y_hat.squeeze()
        y_true = torch.from_numpy(batch[2]).float().to(y_hat.device)
        #loss
        loss_function = nn.BCELoss()
        loss = loss_function(y_hat, y_true)
        #把预测值和真实值分别link在一个大列表里面，用于计算auroc
        self.outPRED_train = torch.cat((self.outPRED_train.to(self.device), y_hat), 0)
        self.outGT_train = torch.cat((self.outGT_train.to(self.device), y_true), 0)
        #tensorboard
        self.log('train_loss', loss)
        train_loss = loss
        return train_loss
    
    def validation_step(self, batch,batch_idx):#定义验证过程每一步
        #读取数据并预处理
        y_hat,_ = self(batch)
        y_hat = y_hat.squeeze()
        y_true = torch.from_numpy(batch[2]).float().to(y_hat.device)
        #loss
        loss_function = nn.BCELoss()
        loss = loss_function(y_hat, y_true)
        #把预测值和真实值分别link在一个大列表里面，用于计算auroc
        self.outPRED_val = torch.cat((self.outPRED_val.to(self.device) , y_hat), 0)
        self.outGT_val  = torch.cat((self.outGT_val.to(self.device) , y_true), 0)
        self.log('val_loss', loss)
        #tensorboard
        val_loss = loss
        return val_loss
    
    def test_step(self, batch, batch_idx):
        y_hat,_ = self(batch)
        y_hat = y_hat.squeeze()
        y_true = torch.from_numpy(batch[2]).float().to(y_hat.device)
        #loss
        loss_function = nn.BCELoss()
        loss = loss_function(y_hat, y_true)
        #把预测值和真实值分别link在一个大列表里面，用于计算auroc
        self.outPRED_test = torch.cat((self.outPRED_val.to(self.device) , y_hat), 0)
        self.outGT_test  = torch.cat((self.outGT_val.to(self.device) , y_true), 0)
        #tensorboard
        test_loss = loss
        return test_loss




    
    def training_epoch_end(self, batch):
        auroc= metrics.roc_auc_score(self.outGT_train.data.cpu().numpy(), self.outPRED_train.data.cpu().numpy())
        auprc= metrics.average_precision_score(self.outGT_train.data.cpu().numpy(), self.outPRED_train.data.cpu().numpy())
        self.log('train_auroc_epoch', auroc)
        self.log('train_auprc_epoch', auprc)
        self.outGT_train = torch.FloatTensor()
        self.outPRED_train = torch.FloatTensor()
    def validation_epoch_end(self, batch):
        auroc= metrics.roc_auc_score(self.outGT_val.data.cpu().numpy(), self.outPRED_val.data.cpu().numpy())
        auprc= metrics.average_precision_score(self.outGT_val.data.cpu().numpy(), self.outPRED_val.data.cpu().numpy())
        self.log('val_auroc_epoch', auroc)
        self.log('val_auprc_epoch', auprc)
        self.outPRED_val = torch.FloatTensor()
        self.outGT_val = torch.FloatTensor()
    def test_epoch_end(self, batch):
        auroc= metrics.roc_auc_score(self.outGT_test.data.cpu().numpy(), self.outPRED_test.data.cpu().numpy())
        auprc= metrics.average_precision_score(self.outGT_test.data.cpu().numpy(), self.outPRED_test.data.cpu().numpy())
        self.log('test_auroc_epoch', auroc)
        self.log('test_auprc_epoch', auprc)
        self.outPRED_test = torch.FloatTensor()
        self.outGT_test = torch.FloatTensor()
        print('Test auroc is: {}'.format(auroc))
        print('Test auprc is: {}'.format(auprc))

    

    def configure_optimizers(self): #这个函数是用来配置优化器的

        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-5,betas=(0.9, 0.9))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=1e-5, patience=4, verbose=True,min_lr=1e-6)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler, "monitor": "val_auroc_epoch"}
