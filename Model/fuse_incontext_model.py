#这个文件中我们准备使用in-context learning来处理missing modality的问题
#基础文件是使用ehr_cxr_partial数据集，然后cxr如果missing的话直接给标志位，先不做特殊处理的
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
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
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
    
class fuse_incontext_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        #-----------------------------------保存的一些参数----------------------------------#
        self.save_hyperparameters() 
        self.max_steps = self.hparams.config.max_steps
        self.task=self.hparams.config.task
        #-------------------------------模型权重加载与初始化--------------------------------#
        #是否加载预训练权重
        if self.hparams.config.load_path  == "":
            self.transformer = getattr(vit, self.hparams.config.vit)(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config.vit )(
                pretrained=False, config=self.hparams.config
            )
        #另外一种加载训练权重
        if (
            self.hparams.config.load_path  != ""
            and not self.hparams.config.test_only 
        ):
            ckpt = torch.load(self.hparams.config.load_path , map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False) 
        #test时候加载检查点 
        if self.hparams.config.load_path_test  != "" and self.hparams.config.test_only :
            ckpt = torch.load(self.hparams.config.load_path_test , map_location="cpu")
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

        #----------------------定义一些属性，在每个step中更新，在每个epoch结束后计算--------------------------------#
        self.outGT_train = torch.FloatTensor()
        self.outPRED_train = torch.FloatTensor()

        self.outGT_val = torch.FloatTensor()
        self.outPRED_val = torch.FloatTensor()

        self.outGT_test = torch.FloatTensor()
        self.outPRED_test = torch.FloatTensor()
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
        #分类器
        if self.hparams.config.precision == 32:
            self.mimic_classifier = nn.Sequential(
                    nn.Linear(hs, hs * 2),
                    nn.LayerNorm(hs * 2),
                    nn.GELU(),
                    nn.Linear(hs * 2, 1),
                    nn.Sigmoid(),
                )
        elif self.hparams.config.precision == 16:
            self.mimic_classifier = nn.Sequential(
                    nn.Linear(hs, hs * 2),
                    nn.LayerNorm(hs * 2),
                    nn.GELU(),
                    nn.Linear(hs * 2, 1),
                    # nn.Sigmoid(),
                )

        self.mimic_classifier.apply(objectives.init_weights)  
        self.pooler = heads.Pooler( config.hidden_size )
        self.pooler.apply(objectives.init_weights)

        #-------------------------missing prompt的设置----------------------------------------------------------------#
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

        if not self.learnt_p:
            self.complete_prompt.requires_grad=False
            self.missing_text_prompt.requires_grad=False           
            self.missing_img_prompt.requires_grad=False


    def forward(self, batch):
        #----------------------------------------------从batch中提取要用的数据----------------------------------------------#
        
        ehr_data=torch.from_numpy(batch[0]).float()
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
            prompts = None
            for idx in range(len(flag_pair)):
                if flag_pair[idx] == 'complete':
                    prompt = self.complete_prompt    #prompt是6*16*768的张量
                    # print('complete')
                elif flag_pair[idx] == 'missed_cxr':
                    prompt = self.missing_img_prompt #prompt是6*16*768的张量
                    image_embeds[idx,1:].fill_(0)    # image_embeds的维度是 128*145*768，就是把第二个元素到最后一个元素全部填充为0
                    # print('missed_cxr')
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

            x = self.transformer.norm(x)

            #----------------------------------------------------处理prompt-----------------------------------------------------#
            if self.prompt_type == 'input':
                total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]
            elif self.prompt_type == 'attention':
                total_prompt_len = prompts.shape[-2]
    
            if self.prompt_type == 'input':
                cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])   
                #cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
            elif self.prompt_type == 'attention':
                cls_feats = self.pooler(x)
            

            #-----------------------------------------------------返回结果-----------------------------------------------------#
            pre = self.mimic_classifier(cls_feats)
            return pre


    def training_step(self, batch, batch_idx):#定义训练过程每一步
        #读取数据并预处理
        y_hat = self(batch)
        y_hat = y_hat.squeeze()
        y_true = torch.from_numpy(batch[2]).float().to(y_hat.device)
        #loss
        if self.hparams.config.precision == 32:
            loss_function = nn.BCELoss()
        elif self.hparams.config.precision == 16:
            loss_function = torch.nn.BCEWithLogitsLoss
        loss = loss_function(y_hat, y_true)
        #tensorboard
        self.log('train_loss', loss)
        # train_loss = loss
        return {'preds': y_hat.detach(), 'targets': y_true.detach(),'loss': loss}
    
    def validation_step(self, batch,batch_idx):#定义验证过程每一步
        #读取数据并预处理
        y_hat= self(batch)
        y_hat = y_hat.squeeze()
        y_true = torch.from_numpy(batch[2]).float().to(y_hat.device)
        #loss
        if self.hparams.config.precision == 32:
            loss_function = nn.BCELoss()
        elif self.hparams.config.precision == 16:
            loss_function = torch.nn.BCEWithLogitsLoss
        loss = loss_function(y_hat, y_true)
        #tensorboard
        self.log('val_loss', loss)
        # val_loss = loss
        return {'preds': y_hat.detach(), 'targets': y_true.detach(),'loss': loss}
    
    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        y_hat = y_hat.squeeze()
        y_true = torch.from_numpy(batch[2]).float().to(y_hat.device)
        #loss
        if self.hparams.config.precision == 32:
            loss_function = nn.BCELoss()
        elif self.hparams.config.precision == 16:
            loss_function = torch.nn.BCEWithLogitsLoss
        loss = loss_function(y_hat, y_true)
        #把预测值和真实值分别link在一个大列表里面，用于计算auroc
        # self.outPRED_test = torch.cat((self.outPRED_val.to(self.device) , y_hat), 0)
        # self.outGT_test  = torch.cat((self.outGT_val.to(self.device) , y_true), 0)
        #tensorboard
        test_loss = loss
        return test_loss


    def training_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])

        auroc= metrics.roc_auc_score(targets.data.cpu().numpy(), preds.data.cpu().numpy())
        auprc= metrics.average_precision_score(targets.data.cpu().numpy(), preds.data.cpu().numpy())
        self.log('train_auroc_epoch', auroc)
        self.log('train_auprc_epoch', auprc)
        # self.outGT_train = torch.FloatTensor()
        # self.outPRED_train = torch.FloatTensor()

        screen = os.environ.get('STY', 'Not available')
        print('server: {} || gpu: No.{} || screen: {}'.format(socket.gethostname(), self.hparams.config.gpu_id, screen))
    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])

        auroc= metrics.roc_auc_score(targets.data.cpu().numpy(), preds.data.cpu().numpy())
        auprc= metrics.average_precision_score(targets.data.cpu().numpy(), preds.data.cpu().numpy())
        self.log('val_auroc_epoch', auroc)
        self.log('val_auprc_epoch', auprc)
        # self.outPRED_val = torch.FloatTensor()
        # self.outGT_val = torch.FloatTensor()
        # screen = os.environ.get('STY', 'Not available')
        # print('server: {} || gpu: No.{} || screen: {}'.format(socket.gethostname(), self.hparams.config.gpu_id, screen))
    def test_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])

        auroc= metrics.roc_auc_score(targets.data.cpu().numpy(), preds.data.cpu().numpy())
        auprc= metrics.average_precision_score(targets.data.cpu().numpy(), preds.data.cpu().numpy())
        self.log('test_auroc_epoch', auroc)
        self.log('test_auprc_epoch', auprc)

        # self.outPRED_test = torch.FloatTensor()
        # self.outGT_test = torch.FloatTensor()
        print('Test auroc is: {}'.format(auroc))
        print('Test auprc is: {}'.format(auprc))

    

    def configure_optimizers(self): #这个函数是用来配置优化器的

        #初始学习率lr,然后经过max_steps个steps下降到lr_end
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.config.learning_rate, eps=1e-8, betas=(0.9, 0.98))
        max_steps = self.hparams.config.max_steps
        decay_power = self.hparams.config.decay_power
        end_lr = self.hparams.config.end_lr

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
