import pytorch_lightning as pl
from Model.tools import tools_ehr, tools_metric,tools_image
import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import skimage, torchvision


class image_only_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters() #这个函数可以直接保存所有的入口参数(config)到checkpoint中    
        self.max_steps = self.hparams.config.max_steps
        self.task=self.hparams.config.task
        self.image_model = tools_image.XRayImageClassifier()


        #定义一些属性，在每个step中更新，在每个epoch结束后计算
        self.outGT_train = torch.FloatTensor()
        self.outPRED_train = torch.FloatTensor()

        self.outGT_val = torch.FloatTensor()
        self.outPRED_val = torch.FloatTensor()

        self.outGT_test = torch.FloatTensor()
        self.outPRED_test = torch.FloatTensor()


    def forward(self, batch):

        #----------------------------------------------从batch中提取要用的数据----------------------------------------------#
        image_data=batch[1]
        #-----------------------------------------------------数据的处理---------------------------------------------------#
        pre_image,feat_image =  self.image_model(image_data)


        #-----------------------------------------------------返回结果-----------------------------------------------------#
        return pre_image,feat_image

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.config.learning_rate,betas=(0.9, 0.9))
        # scheduler = {
        # 'scheduler': CosineAnnealingLR(self.optimizer, T_max=self.max_steps, eta_min=0, last_epoch=-1),
        # 'name': 'cosine_annealing_scheduler',
        # 'interval': 'step',
        # 'frequency': 1
        #             }
        # return {'optimizer': self.optimizer, 'lr_scheduler': scheduler}
        return self.optimizer
