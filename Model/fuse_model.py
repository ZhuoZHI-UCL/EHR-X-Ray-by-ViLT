#这个文件我们使用 ehr_cxr_paired/partial进行训练和测试
import pytorch_lightning as pl
from Model.tools import tools_ehr, tools_metric,tools_image
import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import skimage, torchvision
from Model.tools.tools_fuse.vilt.modules import heads, objectives, vilt_utils
import Model.tools.tools_fuse.vilt.modules.vision_transformer_prompts as vit
from Model.tools import tools_ehr, tools_metric
from torchvision.models import resnet34
from torch.optim.lr_scheduler import LambdaLR
import os
import socket
import shutil
import numpy as np
from transformers import (get_polynomial_decay_schedule_with_warmup,get_cosine_schedule_with_warmup)
from tqdm import tqdm
#将ehr数据截断/填补到统一长度
def pad_or_truncate(data, length_max): 
    batchsize, _, feature = data.shape
    padded_data = np.zeros((batchsize, length_max, feature))

    for i in range(batchsize):
        length_current = len(data[i])
        if length_current <= length_max:
            padded_data[i, :length_current] = data[i]
        else:
            padded_data[i] = data[i, :length_max]
        tensor_data = torch.from_numpy(padded_data).clone()
    
    return tensor_data.float()

#对ehr数据进行平均池化到相同的长度
def mean_pooling(batch, length_max):
    # 获取原始batch的维度
    batch_size, length, feature = batch.shape
    
    # 初始化AdaptiveAvgPool1d
    pool = nn.AdaptiveAvgPool1d(length_max)
    
    # 因为AdaptiveAvgPool1d在最后一个维度上进行操作，所以我们需要调整batch的维度
    batch = batch.permute(0, 2, 1)
    
    # 进行mean pooling
    batch_pooled = pool(batch)
    
    # 将batch的维度调整回原始顺序
    batch_pooled = batch_pooled.permute(0, 2, 1)
    
    return batch_pooled
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_len,dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len , dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

#EHR数据嵌入，
class EHREmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim, max_len):
        super(EHREmbedding, self).__init__()
        self.feature_embedding = nn.Linear(feature_dim,embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len+1)

    def forward(self, x):
        x = x.to(self.feature_embedding.weight.device)
        x = self.feature_embedding(x)

        # 创建 CLS token
        cls_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = self.positional_encoding(x)
        return x
    
class fuse_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        #-----------------------------------保存的一些参数----------------------------------#
        self.save_hyperparameters() 
        self.max_steps = self.hparams.config.max_steps
        self.task=self.hparams.config.task
        #-------------------------------模型权重加载与初始化--------------------------------#
        #是否加载ViT预训练权重
        if self.hparams.config.load_path  == "":
            self.transformer = getattr(vit, self.hparams.config.vit)(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config.vit )(
                pretrained=False, config=self.hparams.config
            )
        #加载预训练权重
        if (
            self.hparams.config.load_path  != ""
            and not self.hparams.config.test_only 
        ):
            ckpt = torch.load(self.hparams.config.load_path , map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False) 

        #fine-tune设置
        for param in self.transformer.parameters():
            param.requires_grad=True

        for param in self.transformer.patch_embed.parameters():
            param.requires_grad = True       
        self.transformer.pos_embed.requires_grad = True     
        self.transformer.cls_token.requires_grad = True
        for param in self.transformer.pos_drop.parameters():
            param.requires_grad = True                      

        #--------------------------------------------EHR嵌入的设置----------------------------------------------#
        self.token_type_embeddings = nn.Embedding(2,  config.hidden_size )
        self.token_type_embeddings.apply(objectives.init_weights)
        self.EHREmbedding=EHREmbedding(self.hparams.config.ehr_feature_size , 
                                            self.hparams.config.hidden_size , 
                                            self.hparams.config.ehr_max_len )
        self.EHREmbedding.apply(objectives.init_weights)       
        #检验单个模态的可行性
        self.ehr_model = tools_ehr.LSTM_Medfuse()
        #--------------------------------------------图像嵌入的设置----------------------------------------------#
        self.image_model = resnet34(pretrained=True)
        num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        #-------------------------下游任务的设置----------------------------------------------------------------#
        hs = self.hparams.config.hidden_size #768 transformer的参数   
        num_classes = self.hparams.config.num_classes #分类的类别数  
        #分类器
        if self.hparams.config.precision == 32:
            self.mimic_classifier = nn.Sequential(
                    nn.Linear(hs, hs * 2),
                    nn.LayerNorm(hs * 2),
                    nn.GELU(),
                    nn.Linear(hs * 2, num_classes),
                    nn.Sigmoid(),
                )
        elif self.hparams.config.precision == 16:
            self.mimic_classifier = nn.Sequential(
                    nn.Linear(hs, hs * 2),
                    nn.LayerNorm(hs * 2),
                    nn.GELU(),
                    nn.Linear(hs * 2, num_classes),
                    # nn.Sigmoid(),
                )

        self.mimic_classifier.apply(objectives.init_weights)  
        self.pooler = heads.Pooler( config.hidden_size )
        self.pooler.apply(objectives.init_weights)

        #-------------------------missing prompt的设置----------------------------------------------------------------#
        if self.hparams.config.missing_prompt:
            #定义关于prompt的参数了
            self.prompt_type = self.hparams.config.prompt_type
            prompt_length = self.hparams.config.prompt_length
            self.prompt_length = prompt_length
            embed_dim = self.hparams.config.hidden_size
            self.learnt_p = self.hparams.config.learnt_p
            self.prompt_layers = self.hparams.config.prompt_layers
            self.multi_layer_prompt = self.hparams.config.multi_layer_prompt
            prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1 #prompt_num=6

            #1.ehr-image完整的情况下
            complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim) #pompt的维度是 6*16*768
            complete_prompt[:,0:1,:].fill_(1)  #把6*1*768的张量全部填充为1，为了指示这是第一种缺失情况          
            if self.learnt_p and self.prompt_type == 'attention':
                complete_prompt[:,prompt_length//2:prompt_length//2+1,:].fill_(1)
            self.complete_prompt = nn.Parameter(complete_prompt)#这句话就是把这个向量设置成可学习的 

            #2. image missing的情况下
            missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
            missing_img_prompt[:,1:2,:].fill_(1)  #把6*1*768的张量全部填充为1，为了指示这是第二种缺失情况           
            if self.learnt_p and self.prompt_type == 'attention':
                missing_img_prompt[:,prompt_length//2+1:prompt_length//2+2,:].fill_(1)
            self.missing_img_prompt = nn.Parameter(missing_img_prompt)

            #3. ehr missing的情况下
            # missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
            # missing_text_prompt[:,2:3,:].fill_(1)            
            # if self.learnt_p and self.prompt_type == 'attention':
            #     missing_text_prompt[:,prompt_length//2+2:prompt_length//2+3,:].fill_(1)
            # self.missing_text_prompt = nn.Parameter(missing_text_prompt)
            
            if not self.learnt_p:
                self.complete_prompt.requires_grad=False
                self.missing_text_prompt.requires_grad=False           
                self.missing_img_prompt.requires_grad=False

        #不同的任务，标签的位置不同
        if self.hparams.config.data_pairs== 'radiology':
            self.position_label = 3
        else: 
            self.position_label = 2
        #----------------------------------------------定义一些计算评价指标需要用到的----------------------------------------------------------------#
        self.all_preds_train = []
        self.all_labels_train = []
        self.complete_preds_train = []
        self.complete_labels_train = []
        self.missed_ehr_preds_train = []
        self.missed_ehr_labels_train = []
        self.missed_cxr_preds_train = []
        self.missed_cxr_labels_train = []
        self.ehr_paired_preds_train = [] #只看当在paired数据中去掉image的情况下的数据
        self.ehr_paired_labels_train = []


        self.all_preds_val = []
        self.all_labels_val = []
        self.complete_preds_val = []
        self.complete_labels_val = []
        self.missed_ehr_preds_val = []
        self.missed_ehr_labels_val = []
        self.missed_cxr_preds_val = []
        self.missed_cxr_labels_val = []
        self.ehr_paired_preds_val = [] #只看当在paired数据中去掉image的情况下的数据
        self.ehr_paired_labels_val = []
        self.best_final_matrix = 0


        self.all_preds_test = []
        self.all_labels_test = []
        self.complete_preds_test = []
        self.complete_labels_test = []
        self.missed_ehr_preds_test = []
        self.missed_ehr_labels_test = []
        self.missed_cxr_preds_test = []
        self.missed_cxr_labels_test = []
        self.ehr_paired_preds_test = [] #只看当在paired数据中去掉image的情况下的数据
        self.ehr_paired_labels_test = []


    
    def forward(self, batch):
        #----------------------------------------------从batch中提取要用的数据----------------------------------------------#
        ehr_data=torch.from_numpy(batch[0]).float()
        if self.hparams.config.task == 'phenotyping':
            #对ehr数据进行池化
            ehr_data = mean_pooling(ehr_data,self.hparams.config.ehr_max_len)

        # ehr_data = tools_ehr.replace_outliers_with_mean(ehr_data)#去一下异常值，使用均值代替
        image_data=batch[1]
        seq_length=torch.tensor(batch[4])
        flag_pair = batch[5]
        #----------------------------------------------模态的选择---------------------------------------------------------#
        test_model = 'fuse'
        if test_model == 'ehr':#比较好的参数 batchsize 256  lr 1e-4 auroc 0.82 NUM 11
            pre_ehr,ehr_embeds =  self.ehr_model(ehr_data,seq_length)
            return pre_ehr
        elif test_model == 'image': #batchsize 128  lr 5e-5 auroc 0.8  NUM 14
            pre_image = self.image_model(image_data)
            return pre_image
        elif test_model == 'fuse': #batchsize 
            #-----------------------------------------------------数据的处理---------------------------------------------------#
            (image_embeds, image_masks, patch_index, image_labels, ) = self.transformer.visual_embed(image_data,
                                                                            max_image_len=self.hparams.config.max_image_len ,
                                                                            mask_it=False,)
            #image_embeds的维度是 batchsize*max_image_len+1*hidden_size
            ehr_embeds =  self.EHREmbedding(ehr_data)

            
            #检测图像是否缺失，并加上prompt,这里我们判断的标准是读取每个batch中第五个元素是否为True
            #如果图像缺失了，则用全0来填补
            if self.hparams.config.missing_prompt:
                prompts = None
                for flag in range(len(flag_pair)):
                    if flag_pair[flag] == 'complete':
                        prompt = self.complete_prompt    #prompt是6*16*768的张量
                    elif flag_pair[flag] == 'missed_cxr':
                        prompt = self.missing_img_prompt #prompt是6*16*768的张量
                        image_embeds[flag,1:].fill_(0)    # image_embeds的维度是 128*145*768，就是把第二个元素到最后一个元素全部填充为0
                    elif flag_pair[flag] == 'missed_ehr':
                        prompt = self.missing_text_prompt
                        ehr_embeds[flag,1:].fill_(0)

                    
                    if prompt.size(0) != 1: 
                        prompt = prompt.unsqueeze(0)
                    
                    if prompts is None:
                        prompts = prompt
                    else:
                        prompts = torch.cat([prompts, prompt], dim=0)           

            # #两个都加token type embedding
            ehr_embeds, image_embeds=( ehr_embeds + self.token_type_embeddings(torch.zeros_like(ehr_embeds[:, :, 0]).long()), 
                                        image_embeds+ self.token_type_embeddings(torch.full_like(image_masks, 1)))

            #---------------------------------------------------连接两个模态的数据---------------------------------------------------#
            co_embeds = torch.cat([ehr_embeds, image_embeds], dim=1)
            x=co_embeds
            # x = co_embeds.detach()
            if self.hparams.config.missing_prompt:
                for i, blk in enumerate(self.transformer.blocks):
                    if i in self.prompt_layers:#需要加prompt的层会在config里面定义
                        if self.multi_layer_prompt:
                            x, _attn = blk(x, 
                                        prompts=prompts[:,self.prompt_layers.index(i)], 
                                        learnt_p=self.learnt_p,
                                        prompt_type=self.prompt_type)
                        else:
                            x, _attn = blk(x, prompts=prompts, learnt_p=self.learnt_p)
                    else:
                        x, _attn = blk(x)
            else:
                for i, blk in enumerate(self.transformer.blocks):
                    x, _attn = blk(x)

            x = self.transformer.norm(x) #batchsize*(49+145+96)*768

            #----------------------------------------------------处理prompt-----------------------------------------------------#
            if self.hparams.config.missing_prompt:
                if self.prompt_type == 'input':
                    total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]
                elif self.prompt_type == 'attention':
                    total_prompt_len = prompts.shape[-2]
        
                if self.prompt_type == 'input':
                    cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])   
                    #cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
                elif self.prompt_type == 'attention':
                    cls_feats = self.pooler(x)
            else:
                cls_feats = self.pooler(x)

            #-----------------------------------------------------返回结果-----------------------------------------------------#
            pre = self.mimic_classifier(cls_feats)
            return pre

    #把验证集直接放在GPU上
    def setup(self, stage):
        if self.hparams.config.preload_val:
            if stage == 'fit':
                self.preload_val_data()


    def preload_val_data(self):
        self.preloaded_val_data = []
        dataloader = self.trainer.datamodule.val_dataloader()
        
        for batch in tqdm(dataloader, desc="Loading validation data"):
            batch = move_to_device(batch, self.device)
            self.preloaded_val_data.append(batch)
    
    def move_to_device(batch, device):
        if torch.is_tensor(batch):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [move_to_device(v, device) for v in batch]
        else:
            return batch 
    def training_step(self, batch, batch_idx):#定义训练过程每一步
        #预测
        y_hat = self(batch)
        y_hat = y_hat.squeeze()
        #真实标签
        if self.hparams.config.data_pairs== 'radiology':
            y_true = batch[self.position_label].float().to(y_hat.device)
        else:
            y_true = torch.from_numpy(batch[self.position_label]).float().to(y_hat.device)
        #分割每个类别的预测与标签
        self.all_preds_train.append(y_hat)
        self.all_labels_train.append(y_true)
        self.complete_preds_train.append(y_hat[torch.tensor([x == "complete" for x in batch[5]])]) #batch[5]是list
        self.complete_labels_train.append(y_true[torch.tensor([x == "complete" for x in batch[5]])])
        self.missed_cxr_preds_train.append(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])
        self.missed_cxr_labels_train.append(y_true[torch.tensor([x == "missed_cxr" for x in batch[5]])])
        self.missed_ehr_preds_train.append(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])
        self.missed_ehr_labels_train.append(y_true[torch.tensor([x == "missed_ehr" for x in batch[5]])])   


        #loss
        if self.hparams.config.precision == 32:
            loss_function = nn.BCELoss()
        elif self.hparams.config.precision == 16:
            loss_function = torch.nn.BCEWithLogitsLoss
        if self.hparams.config.reweight_loss: #调整一下loss哈
            if len(y_hat[torch.tensor([x == "complete" for x in batch[5]])])>0:
                complete_loss = loss_function(y_hat[torch.tensor([x == "complete" for x in batch[5]])], y_true[torch.tensor([x == "complete" for x in batch[5]])])
                # complete_weight = 1.0 / len(y_hat[torch.tensor([x == "complete" for x in batch[5]])])
            else:
                complete_loss = 0
                complete_weight = 0
            
            if len(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])>0:
                missed_cxr_loss = loss_function(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])], y_true[torch.tensor([x == "missed_cxr" for x in batch[5]])])
                # missed_cxr_weight = 1.0 / len(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])
            else:
                missed_cxr_loss = 0
                missed_cxr_weight = 0
            
            if len(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])>0:
                missed_ehr_loss = loss_function(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])], y_true[torch.tensor([x == "missed_ehr" for x in batch[5]])])
                # missed_ehr_weight = 1.0 / len(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])
            else:
                missed_ehr_loss = 0
                missed_ehr_weight = 0


            # Compute the total loss as the weighted sum of the individual losses
            # loss = complete_weight * complete_loss + missed_cxr_weight * missed_cxr_loss + missed_ehr_weight * missed_ehr_loss
            loss = 3 * complete_loss +  missed_cxr_loss 

        else:
            loss = loss_function(y_hat, y_true)

        #tensorboard
        self.log('train_loss', loss)
        return {'preds': y_hat.detach(), 'targets': y_true.detach(),'loss': loss}
    
    def validation_step(self, batch,batch_idx):#定义验证过程每一步
        #预测
        if self.hparams.config.preload_val:
            batch = self.preloaded_val_data[batch_idx]
        y_hat= self(batch)
        y_hat = y_hat.squeeze()
        #真实标签
        if self.hparams.config.data_pairs== 'radiology':
            y_true = batch[self.position_label].float().to(y_hat.device)
        else:
            y_true = torch.from_numpy(batch[self.position_label]).float().to(y_hat.device)
        #分割每个类别的预测与标签
        self.all_preds_val.append(y_hat) #y_hat是tensor
        self.all_labels_val.append(y_true)
        self.complete_preds_val.append(y_hat[torch.tensor([x == "complete" for x in batch[5]])]) #batch[5]是list
        self.complete_labels_val.append(y_true[torch.tensor([x == "complete" for x in batch[5]])])
        self.missed_cxr_preds_val.append(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])
        self.missed_cxr_labels_val.append(y_true[torch.tensor([x == "missed_cxr" for x in batch[5]])])
        self.missed_ehr_preds_val.append(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])
        self.missed_ehr_labels_val.append(y_true[torch.tensor([x == "missed_ehr" for x in batch[5]])])
        #loss
        if self.hparams.config.precision == 32:
            loss_function = nn.BCELoss()
        elif self.hparams.config.precision == 16:
            loss_function = torch.nn.BCEWithLogitsLoss
        
        if self.hparams.config.reweight_loss: #调整一下loss哈
            if len(y_hat[torch.tensor([x == "complete" for x in batch[5]])])>0:
                complete_loss = loss_function(y_hat[torch.tensor([x == "complete" for x in batch[5]])], y_true[torch.tensor([x == "complete" for x in batch[5]])])
                # complete_weight = 1.0 / len(y_hat[torch.tensor([x == "complete" for x in batch[5]])])
            else:
                complete_loss = 0
                complete_weight = 0
            
            if len(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])>0:
                missed_cxr_loss = loss_function(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])], y_true[torch.tensor([x == "missed_cxr" for x in batch[5]])])
                # missed_cxr_weight = 1.0 / len(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])
            else:
                missed_cxr_loss = 0
                missed_cxr_weight = 0
            
            if len(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])>0:
                missed_ehr_loss = loss_function(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])], y_true[torch.tensor([x == "missed_ehr" for x in batch[5]])])
                # missed_ehr_weight = 1.0 / len(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])
            else:
                missed_ehr_loss = 0
                missed_ehr_weight = 0


            # Compute the total loss as the weighted sum of the individual losses
            # loss = complete_weight * complete_loss + missed_cxr_weight * missed_cxr_loss + missed_ehr_weight * missed_ehr_loss
            loss = 3 * complete_loss + missed_cxr_loss 

        else:
            loss = loss_function(y_hat, y_true)
        #tensorboard
        self.log('val_loss', loss)
        return {'preds': y_hat.detach(), 'targets': y_true.detach(),'loss': loss}
    
    def test_step(self, batch, batch_idx):
        #预测
        y_hat= self(batch)
        y_hat = y_hat.squeeze()
        #真实标签
        if self.hparams.config.data_pairs== 'radiology':
            y_true = batch[self.position_label].float().to(y_hat.device)
        else:
            y_true = torch.from_numpy(batch[self.position_label]).float().to(y_hat.device)
        #分割每个类别的预测与标签
        self.all_preds_test.append(y_hat) #y_hat是tensor
        self.all_labels_test.append(y_true)
        self.complete_preds_test.append(y_hat[torch.tensor([x == "complete" for x in batch[5]])]) #batch[5]是list
        self.complete_labels_test.append(y_true[torch.tensor([x == "complete" for x in batch[5]])])
        self.missed_cxr_preds_test.append(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])
        self.missed_cxr_labels_test.append(y_true[torch.tensor([x == "missed_cxr" for x in batch[5]])])
        self.missed_ehr_preds_test.append(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])
        self.missed_ehr_labels_test.append(y_true[torch.tensor([x == "missed_ehr" for x in batch[5]])])
        #loss
        if self.hparams.config.precision == 32:
            loss_function = nn.BCELoss()
        elif self.hparams.config.precision == 16:
            loss_function = torch.nn.BCEWithLogitsLoss
        
        if self.hparams.config.reweight_loss: #调整一下loss哈
            if len(y_hat[torch.tensor([x == "complete" for x in batch[5]])])>0:
                complete_loss = loss_function(y_hat[torch.tensor([x == "complete" for x in batch[5]])], y_true[torch.tensor([x == "complete" for x in batch[5]])])
                # complete_weight = 1.0 / len(y_hat[torch.tensor([x == "complete" for x in batch[5]])])
            else:
                complete_loss = 0
                complete_weight = 0
            
            if len(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])>0:
                missed_cxr_loss = loss_function(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])], y_true[torch.tensor([x == "missed_cxr" for x in batch[5]])])
                # missed_cxr_weight = 1.0 / len(y_hat[torch.tensor([x == "missed_cxr" for x in batch[5]])])
            else:
                missed_cxr_loss = 0
                missed_cxr_weight = 0
            
            if len(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])>0:
                missed_ehr_loss = loss_function(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])], y_true[torch.tensor([x == "missed_ehr" for x in batch[5]])])
                # missed_ehr_weight = 1.0 / len(y_hat[torch.tensor([x == "missed_ehr" for x in batch[5]])])
            else:
                missed_ehr_loss = 0
                missed_ehr_weight = 0


            # Compute the total loss as the weighted sum of the individual losses
            # loss = complete_weight * complete_loss + missed_cxr_weight * missed_cxr_loss + missed_ehr_weight * missed_ehr_loss
            loss = 3 * complete_loss +  missed_cxr_loss 

        else:
            loss = loss_function(y_hat, y_true)
        #tensorboard
        self.log('test_loss', loss)
        return {'preds': y_hat.detach(), 'targets': y_true.detach(),'loss': loss}


    def training_epoch_end(self, outputs):
        # preds = torch.cat([tmp['preds'] for tmp in outputs])
        # targets = torch.cat([tmp['targets'] for tmp in outputs])

        all_auroc = metrics.roc_auc_score( torch.cat(self.all_labels_train).data.cpu().numpy(),torch.cat(self.all_preds_train).data.cpu().numpy())
        all_auprc = metrics.average_precision_score(torch.cat(self.all_labels_train).data.cpu().numpy(),torch.cat(self.all_preds_train).data.cpu().numpy() )
        self.log('train_auroc/train_all_auroc', all_auroc)
        self.log('train_auprc/train_all_auprc', all_auprc)
        #对于missed_cxr我们只检测和paired数量相同的哈（看一下结果怎么样）
        if len(torch.cat(self.complete_preds_train).data.cpu().numpy()) > 0:
            complete_auroc = metrics.roc_auc_score( torch.cat(self.complete_labels_train).data.cpu().numpy(),torch.cat(self.complete_preds_train).data.cpu().numpy())
            complete_auprc = metrics.average_precision_score( torch.cat(self.complete_labels_train).data.cpu().numpy(),torch.cat(self.complete_preds_train).data.cpu().numpy())
            self.log('train_auroc/train_complete_auroc', complete_auroc)
            self.log('train_auprc/train_complete_auprc', complete_auprc)
        
        if len(torch.cat(self.missed_cxr_preds_train).data.cpu().numpy()) > 0:
            missed_cxr_auroc = metrics.roc_auc_score( torch.cat(self.missed_cxr_labels_train).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_train).data.cpu().numpy())
            missed_cxr_auprc = metrics.average_precision_score( torch.cat(self.missed_cxr_labels_train).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_train).data.cpu().numpy())
            self.log('train_auroc/train_missed_cxr_auroc', missed_cxr_auroc)
            self.log('train_auprc/train_missed_cxr_auprc', missed_cxr_auprc)

        if len(torch.cat(self.missed_ehr_preds_train).data.cpu().numpy()) > 0:
            missed_ehr_auroc = metrics.roc_auc_score( torch.cat(self.missed_ehr_labels_train).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_train).data.cpu().numpy())
            missed_ehr_auprc = metrics.average_precision_score( torch.cat(self.missed_ehr_labels_train).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_train).data.cpu().numpy())
            self.log('train_auroc/train_missed_ehr_auroc', missed_ehr_auroc)
            self.log('train_auprc/train_missed_ehr_auprc', missed_ehr_auprc)
        
        final_matrix = all_auroc + all_auprc
        self.log('train_auroc/train_final_matrix', final_matrix)
        #清空列表以便下一轮验证
        self.all_preds_train = []
        self.all_labels_train = []
        self.complete_preds_train = []
        self.complete_labels_train = []
        self.missed_cxr_preds_train = []
        self.missed_cxr_labels_train = []
        self.missed_ehr_preds_train = []
        self.missed_ehr_labels_train = []

        screen = os.environ.get('STY', 'Not available')
        print('server: {} || gpu: No.{} || screen: {}'.format(socket.gethostname(), self.hparams.config.gpu_id, screen))
    def validation_epoch_end(self, outputs):
        # preds = torch.cat([tmp['preds'] for tmp in outputs])
        # targets = torch.cat([tmp['targets'] for tmp in outputs])
        all_auroc = metrics.roc_auc_score( torch.cat(self.all_labels_val).data.cpu().numpy(),torch.cat(self.all_preds_val).data.cpu().numpy())
        all_auprc = metrics.average_precision_score(torch.cat(self.all_labels_val).data.cpu().numpy(),torch.cat(self.all_preds_val).data.cpu().numpy() )
        self.log('val_auroc/val_all_auroc', all_auroc)
        self.log('val_auprc/val_all_auprc', all_auprc)

        if len(torch.cat(self.complete_preds_val).data.cpu().numpy()) > 0:
            complete_auroc = metrics.roc_auc_score( torch.cat(self.complete_labels_val).data.cpu().numpy(),torch.cat(self.complete_preds_val).data.cpu().numpy())
            complete_auprc = metrics.average_precision_score( torch.cat(self.complete_labels_val).data.cpu().numpy(),torch.cat(self.complete_preds_val).data.cpu().numpy())
            self.log('val_auroc/val_complete_auroc', complete_auroc)
            self.log('val_auprc/val_complete_auprc', complete_auprc)
        
        if len(torch.cat(self.missed_cxr_preds_val).data.cpu().numpy()) > 0:
            missed_cxr_auroc = metrics.roc_auc_score( torch.cat(self.missed_cxr_labels_val).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_val).data.cpu().numpy())
            missed_cxr_auprc = metrics.average_precision_score(torch.cat(self.missed_cxr_labels_val).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_val).data.cpu().numpy() )
            self.log('val_auroc/val_missed_cxr_auroc', missed_cxr_auroc)
            self.log('val_auprc/val_missed_cxr_auprc', missed_cxr_auprc)

        if len(torch.cat(self.missed_ehr_preds_val).data.cpu().numpy()) > 0:
            missed_ehr_auroc = metrics.roc_auc_score( torch.cat(self.missed_ehr_labels_val).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_val).data.cpu().numpy())
            missed_ehr_auprc = metrics.average_precision_score( torch.cat(self.missed_ehr_labels_val).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_val).data.cpu().numpy())
            self.log('val_auroc/val_missed_ehr_auroc', missed_ehr_auroc)
            self.log('val_auprc/val_missed_ehr_auprc', missed_ehr_auprc)
        
        current_final_matrix = all_auroc + all_auprc
        self.log('val_auroc/val_final_matrix', current_final_matrix)
        #判断该次final_matrix是否大于上次的，如果大于则更新best_final_matrix,
        #并把 auroc_all, auprc_all, auroc_complete, auprc_complete, auroc_missed_cxr, auprc_missed_cxr, auroc_missed_ehr, auprc_missed_ehr
        #写入到path的txt中
        if current_final_matrix > self.best_final_matrix:
            self.best_final_matrix = current_final_matrix
            with open(f'{self.trainer.logger.log_dir}' + '/best_result.txt', 'w') as f:
                f.write('auroc_all: ' + str(all_auroc if 'all_auroc' in locals() else 'None') + '\n')
                f.write('auprc_all: ' + str(all_auprc if 'all_auprc' in locals() else 'None') + '\n')
                f.write('auroc_complete: ' + str(complete_auroc if 'complete_auroc' in locals() else 'None') + '\n')
                f.write('auprc_complete: ' + str(complete_auprc if 'complete_auprc' in locals() else 'None') + '\n')
                f.write('auroc_missed_cxr: ' + str(missed_cxr_auroc if 'missed_cxr_auroc' in locals() else 'None') + '\n')
                f.write('auprc_missed_cxr: ' + str(missed_cxr_auprc if 'missed_cxr_auprc' in locals() else 'None') + '\n')
                f.write('auroc_missed_ehr: ' + str(missed_ehr_auroc if 'missed_ehr_auroc' in locals() else 'None') + '\n')
                f.write('auprc_missed_ehr: ' + str(missed_ehr_auprc if 'missed_ehr_auprc' in locals() else 'None') + '\n')


        #清空列表以便下一轮验证
        self.all_preds_val = []
        self.all_labels_val = []
        self.complete_preds_val = []
        self.complete_labels_val = []
        self.missed_cxr_preds_val = []
        self.missed_cxr_labels_val = []
        self.missed_ehr_preds_val = []
        self.missed_ehr_labels_val = []

    def test_epoch_end(self, outputs):
        all_auroc = metrics.roc_auc_score(torch.cat(self.all_labels_test).data.cpu().numpy(),torch.cat(self.all_preds_test).data.cpu().numpy())
        all_auprc = metrics.average_precision_score(torch.cat(self.all_labels_test).data.cpu().numpy(),torch.cat(self.all_preds_test).data.cpu().numpy() )
        print(f'auroc_all is: {all_auroc:.3f}')
        print(f'auprc_all is: {all_auprc:.3f}')
        if len(torch.cat(self.complete_preds_test).data.cpu().numpy()) > 0:
            #这个里面我们要做一些工作，把paired的数据中去掉
            complete_auroc = metrics.roc_auc_score( torch.cat(self.complete_labels_test).data.cpu().numpy(),torch.cat(self.complete_preds_test).data.cpu().numpy())
            complete_auprc = metrics.average_precision_score( torch.cat(self.complete_labels_test).data.cpu().numpy(),torch.cat(self.complete_preds_test).data.cpu().numpy())
            print(f'complete_auroc is: {complete_auroc:.3f}')
            print(f'complete_auprc is: {complete_auprc:.3f}')
            print('The number of complete is {}'.format(len(torch.cat(self.complete_preds_test).data.cpu().numpy())))
        if len(torch.cat(self.missed_cxr_preds_test).data.cpu().numpy()) > 0:
            missed_cxr_auroc = metrics.roc_auc_score( torch.cat(self.missed_cxr_labels_test).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_test).data.cpu().numpy())
            missed_cxr_auprc = metrics.average_precision_score(torch.cat(self.missed_cxr_labels_test).data.cpu().numpy(),torch.cat(self.missed_cxr_preds_test).data.cpu().numpy())
            print(f'auroc_missed_cxr is: {missed_cxr_auroc:.3f}')
            print(f'auprc_missed_cxr is: {missed_cxr_auprc:.3f}')
            print('The number of missed_cxr is {}'.format(len(torch.cat(self.missed_cxr_labels_test).data.cpu().numpy())))
    
        if len(torch.cat(self.missed_ehr_preds_test).data.cpu().numpy()) > 0:
            missed_ehr_auroc = metrics.roc_auc_score( torch.cat(self.missed_ehr_labels_test).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_test).data.cpu().numpy())
            missed_ehr_auprc = metrics.average_precision_score( torch.cat(self.missed_ehr_labels_test).data.cpu().numpy(),torch.cat(self.missed_ehr_preds_test).data.cpu().numpy())
            print(f'auroc_missed_ehr is: {missed_ehr_auroc:.3f}')
            print(f'auprc_missed_ehr is: {missed_ehr_auprc:.3f}')           

    def configure_optimizers(self): #这个函数是用来配置优化器的

        #初始学习率lr,然后经过max_steps个steps下降到lr_end
        max_steps = self.hparams.config.max_steps
        decay_power = self.hparams.config.decay_power
        lr = self.hparams.config.learning_rate
        end_lr = self.hparams.config.end_lr
        wd = self.hparams.config.weight_decay
        lr_mult = self.hparams.config.lr_mult
        
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
            ]
        head_names = ["cxr_classifier", "mimic_classifier", "food101_classifier", "hatememes_classifier", "nlvr2_classifier"]
        prompt_name = "prompt"
        names = [n for n, p in self.named_parameters()]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },            
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult,
            },
        ]
        if self.hparams.config.optim_type == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
            )
        warmup_steps = self.hparams.config.warmup_steps
        if isinstance(self.hparams.config.warmup_steps, float):
            warmup_steps = int(max_steps * warmup_steps)

        if decay_power == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        else:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=end_lr,
                power=decay_power,
            )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )
    
    # def configure_optimizers(self): #这个函数是用来配置优化器的

    #     #初始学习率lr,然后经过max_steps个steps下降到lr_end
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.config.learning_rate, eps=1e-8, betas=(0.9, 0.98))
    #     max_steps = self.hparams.config.max_steps
    #     decay_power = self.hparams.config.decay_power
    #     end_lr = self.hparams.config.end_lr

    #     warmup_steps = self.hparams.config.warmup_steps
    #     if isinstance(self.hparams.config.warmup_steps, float):
    #         warmup_steps = int(max_steps * warmup_steps)

    #     if decay_power == "cosine":
    #         scheduler = get_cosine_schedule_with_warmup(
    #             optimizer,
    #             num_warmup_steps=warmup_steps,
    #             num_training_steps=max_steps,
    #         )
    #     else:
    #         scheduler = get_polynomial_decay_schedule_with_warmup(
    #             optimizer,
    #             num_warmup_steps=warmup_steps,
    #             num_training_steps=max_steps,
    #             lr_end=end_lr,
    #             power=decay_power,
    #         )

    #     sched = {"scheduler": scheduler, "interval": "step"}

    #     return (
    #         [optimizer],
    #         [sched],
    #     )
    

    #在每次训练开始的时候干点事情,把重要的文件给复制到训练保存东西的文件夹中
    def on_train_start(self):
        #保存重要的文件设置到对应的result目录下
        files_to_copy = {
            'fuse_config.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_mis/Config/fuse_config.py',
            'Extract_data_1.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_mis/Dataloader/Extract_data_1.py',
            'fuse_model.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_mis/Model/fuse_model.py',
            'run.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_mis/run.py',
        }
        for file_name, source_path in files_to_copy.items():
            target_file_path = f'{self.trainer.logger.log_dir}/{file_name}'
            if not os.path.exists(target_file_path):
                shutil.copyfile(source_path, target_file_path)

        #打印输出环境设置
        # print('server: {} || gpu: No.{} || screen: {}'.format(socket.gethostname(), self.hparams.config.gpu_id, os.environ['STY']))
