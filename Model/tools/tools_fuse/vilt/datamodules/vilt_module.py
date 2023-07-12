import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
import numpy as np
import csv
import os
import shutil
def replace_outliers_with_zero(x):
    mask = (x < 100) | (x > -100)
    x[mask] = 0
    return x
def check_and_print_nan(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f'{tensor_name} contains NaN values.')
#如果我们不用文本的话，需要自己写一个时间序列的嵌入
#位置嵌入,请记住我们要在前面加一个 CLS分类头哈,CLS的位置编码是在第一个
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

import torch
from torch import nn

class LstmEmbedder(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, num_layers, output_token_num, output_token_dim, bidirectional=False):
        super(LstmEmbedder, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(lstm_hidden_dim * (2 if bidirectional else 1), output_token_num * output_token_dim)
        self.output_token_num = output_token_num
        self.output_token_dim = output_token_dim
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch_size, seq_len, num_directions * hidden_size)
        
        # Use the final LSTM hidden state as input to the fully connected layer
        output = self.fc(lstm_out[:, -1, :])
        # output: (batch_size, output_token_num * output_token_dim)

        # Reshape the output to the desired shape
        output = output.view(-1, self.output_token_num, self.output_token_dim)
        # output: (batch_size, output_token_num, output_token_dim)

        return output


#EHR数据嵌入，
class EHREmbedding(nn.Module):
    def __init__(self,ehr_feature_size, output_token_dim,max_legnth):
        super(EHREmbedding, self).__init__()
        self.feature_embedding = nn.Linear(ehr_feature_size,output_token_dim)
        self.positional_encoding = PositionalEncoding(output_token_dim, max_legnth+1)

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        x = x.to(self.feature_embedding.weight.device)
        x = self.feature_embedding(x)

        # 创建 CLS token
        cls_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        # x: (batch_size, seq_len, embed_dim)
        # x = x.transpose(0, 1) #这行代码是将输入张量 x 的第0维（通常对应批次大小）和第1维（通常对应序列长度）进行交换。
                              #这是因为在很多情况下，Transformer的实现需要输入的数据的形状为 (seq_len, batch_size, feature_dim)，
                              # 而不是 (batch_size, seq_len, feature_dim)。这样做的一个主要原因是可以方便地处理不同长度的序列。
        # x: (seq_len, batch_size, embed_dim)
        x = self.positional_encoding(x)
        # x: (seq_len, batch_size, embed_dim)
        return x

# class ResidualClassifier(nn.Module):
#     def __init__(self, hs, cls_num):
#         super().__init__()
#         self.residual_mapping = nn.Linear(hs, hs * 2)
#         self.mimic_mid = nn.Sequential(
#             nn.Linear(hs, hs * 2),
#             nn.BatchNorm1d(hs * 2),
#             # nn.Tanh(),
#         )
#         self.final= nn.Linear(hs*2,cls_num)

#     def forward(self, x):
#         residual = self.residual_mapping(x)
#         x = self.mimic_mid(x)
#         output=self.final(x+residual)
#         return output
#     def get_last_layer(self):
#         return self.final



class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()    
        #是否加载预训练权重
        if self.hparams.config.load_path  == "":
            self.transformer = getattr(vit, self.hparams.config.vit)(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config.vit )(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler( config.hidden_size )#transformer的池化层，用于将 Transformer 编码器的输出特征进行池化，
                                                       #得到一个固定长度的特征向量，用于后续的分类或回归任务
        self.pooler.apply(objectives.init_weights)#一个初始化函数，用于初始化模型参数的权重和偏置，以便模型可以更快地收敛。
        
        self.token_type_embeddings = nn.Embedding(2,  config.hidden_size )
        self.token_type_embeddings.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (##还是加载预训练权重
            self.hparams.config.load_path  != ""
            and not self.hparams.config.test_only 
        ):
            ckpt = torch.load(self.hparams.config.load_path , map_location="cpu")
            state_dict = ckpt["state_dict"]#state_dict是ckpt这个权重的一个键，它是一个字典，用于存储模型的权重和偏置等参数
            self.load_state_dict(state_dict, strict=False) #将给定的状态字典 state_dict 加载到当前模型中，并更新模型的权重和偏置等参数。

        hs = self.hparams.config.hidden_size #768 transformer的参数

        #我们使用的是多分类
        cls_num = self.hparams.config.class_num
        self.mimic_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.BatchNorm1d(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
                # nn.Sigmoid(),
            )
        self.mimic_classifier.apply(objectives.init_weights)
        # self.mimic_classifier = ResidualClassifier(hs, cls_num)
        # self.mimic_classifier.apply(objectives.init_weights)  



        #我们的EHRembedding层也应该是随机初始化参数

        self.EHREmbedding=EHREmbedding(self.hparams.config.ehr_feature_size , 
                                       self.hparams.config.hidden_size,
                                       self.hparams.config.ehr_max_len)
        self.EHREmbedding.apply(objectives.init_weights)


        # ===================== 冻结住原始ViLT的中间层，fine tuning输入层和输出层 ======================
        for param in self.transformer.parameters():
            param.requires_grad=False
        

        #下面都finetuning
        #我不知道这个地方冻结住吗
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=True           

        #我同时也想fine tuning一下图像的嵌入层
        # Unfreeze the parameters of the modules
        for param in self.transformer.patch_embed.parameters():
            param.requires_grad = True

        self.transformer.pos_embed.requires_grad = True   #这个会导致报错

        self.transformer.cls_token.requires_grad = True

        # for param in self.transformer.pos_drop.parameters():
        #     param.requires_grad = True

        for param in self.pooler.parameters():
            param.requires_grad = True
        




        vilt_utils.set_metrics(self)
        self.current_tasks = list()




        # ===================== load downstream (test_only) ======================
        

        if self.hparams.config.load_path_test  != "" and self.hparams.config.test_only :
            ckpt = torch.load(self.hparams.config.load_path_test , map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)


    #----------------------------------------前向传播过程-----------------------------------------#
    def infer( #实际上infer()在后面的调用中只调用了 self.infer(batch) batch一个参数，其他的都是默认参数
        self,
        batch, #batch 就是forward函数中的x，batch的格式是  x, img, y_ehr, y_cxr, seq_lengths, pairs
        # mask_ehr=False,#mask_text用于决定是否对输入的文本进行掩码操作,用于mlm
        mask_image=False,#mask_image决定是否 随机掩盖输入图像中的一部分，然后让模型预测被掩盖的部分。
        image_token_type_idx=1, #区分这两种类型的输入
        image_embeds=None, #输入的图像的嵌入向量
        image_masks=None, #用来指示图像中被遮掩住的地方（1和0来区分，掩码矩阵
    ):  
        
        #输入的数据
        #后续这一步看看能不能在制作dataloader时就给完成了，在这里每次都要调用会耗时间

        ehr_norm=replace_outliers_with_zero(batch[0])

        padding_size=self.hparams.config.ehr_max_len - ehr_norm.shape[1]
        if padding_size > 0:
            # 在seq_len维度进行padding
            ehr_norm= torch.from_numpy(ehr_norm).float()
            ehr_data = F.pad(ehr_norm, (0, 0, 0, padding_size))

        else:
            ehr_data=torch.from_numpy(ehr_norm[:, :self.hparams.config.ehr_max_len, :]).float()

        # ehr_data=torch.from_numpy(replace_outliers_with_zero(batch[0][:, :self.hparams.config.ehr_max_len, :])).float()#16*590*76  float 32(应该是会padding)
        cxr_data=batch[1] # ([16, 3, 384, 384]) float 32
        # print(torch.max(cxr_data[0]))
        # print(torch.mean(cxr_data[0]))
        # print(torch.min(cxr_data[0]))
        ehr_label= torch.from_numpy(batch[2]).int() # (16, 25) 'int32'
    



        # 检查是否有NaN值
        # check_and_print_nan(ehr_data, 'ehr_data')
        # check_and_print_nan(cxr_data, 'cxr_data')
        # check_and_print_nan(ehr_label, 'ehr_label')
        # cxr_label=batch[3]
        # ehr_length=batch[4]
        # flag_pair=batch[5]



        #它看起来是在用 image_token_type_idx 来选择具体要处理的图像。
        # 如果 image_token_type_idx 为 1，那么它会寻找 batch 中的 "image_0"；
        # 如果 image_token_type_idx 为 2，那么它会寻找 "image_1"，
        # 以此类推。如果找不到对应的图像，就会默认使用 "image"。
        # if f"image_{image_token_type_idx - 1}" in batch:
        #     imgkey = f"image_{image_token_type_idx - 1}"
        # else:
        #     imgkey = "image"

    
        #----------------------------处理EHR的区域----------------------------#
        ehr_embeds =  self.EHREmbedding(ehr_data) #torch.Size([16, 590, 768]),float 32
        

         #把原始数据和embedded的数据写入txt对比
        # what_we_want=ehr_embeds
        # what_we_want =  what_we_want.cpu().detach().numpy()

        # file = open('/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/embedding.txt', "a")
        # for sample_embedded, sample_original in zip(what_we_want, original_data):
        #     file.write(str(sample_embedded.max()) 
        #                + "  " + str(sample_embedded.mean()) 
        #                + "  " + str(sample_embedded.var()) 
        #                + "  " + str(sample_embedded.min())
        #                + "  " + str(sample_original.max()) 
        #                + "  " + str(sample_original.mean()) 
        #                + "  " + str(sample_original.var()) 
        #                + "  " + str(sample_original.min())           
        #                )

        #     file.write("\n")
        # file.close()
        
        #把原始数据和embedded的数据写入csv对比
        # filename = '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/embedding.csv'
        # with open(filename, 'a', newline='') as file:
        #     writer = csv.writer(file)
            
        #     for sample_embedded, sample_original in zip(what_we_want, original_data):
        #         row = [
        #             sample_embedded.max(), sample_embedded.mean(), sample_embedded.var(), sample_embedded.min(),
        #             sample_original.max(), sample_original.mean(), sample_original.var(), sample_original.min()
        #         ]
                
        #         writer.writerow(row)
        

        
        # np.savetxt("/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/ehr_demo.csv", what_we_want, delimiter=",", fmt='%.1f')

        
        #----------------------------处理图像的区域----------------------------#
        if image_embeds is None and image_masks is None:#这两个是设置成了none，所以只用执行第一个，就是直接
            img = cxr_data                     #用vit的嵌入来嵌入图像
            (
                image_embeds, #torch.Size([16, 145, 768]),我他妈才知道，每个块是32*32，划分了12*12个块，第一个应该是class头
                image_masks, # ([16, 145]) 每个sample的145个都是1
                patch_index, # len(patch_index)=2  len(patch_index[0])=16 len(patch_index[0][0]) = 144
                image_labels, # None
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config.max_image_len ,
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        #请注意，图像本来是有




        #-----------------------------处理两个模态一起的embedding-------------------------#
        ehr_embeds, image_embeds = ( #这一步是给两个嵌入的模态分别加上模态标识
                                    #token_type_embeddings接收一个和输入嵌入向量同样大小的全0或全1的tensor，
                                    # 然后生成一个同样大小的嵌入向量，这个向量在加到原始的输入嵌入向量上，
                                    # 就起到了区分不同类型输入的作用。
            ehr_embeds + self.token_type_embeddings(torch.zeros_like(ehr_embeds[:, :, 0]).long()), #我们没有text_masks，就直接使用ehr_embeds生成
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )


        #ehr_embeds([8, 387, 768]),image_embeds([8, 257, 768])
        co_embeds = torch.cat([ehr_embeds, image_embeds], dim=1)#co_embeds就包含了输入的所有信息，即文本和图像的嵌入信息。
        #这里我们人工生成一个和原vit中  text_masks维度一样但是全是1的向量,image_masks([8, 257]),基本全是1

        #不考虑mask
        # co_masks = torch.cat([torch.ones(ehr_embeds[:, :, 0].shape).to(image_masks.device), image_masks], dim=1)#text_masks和image_masks也在维度1上进行合并，得到co_masks。
        #这个co_masks对应于co_embeds，用来指示co_embeds中哪些位置的数据是有效的，哪些位置的数据是被掩盖的或者是无效的。

        x = co_embeds #torch.Size([16, 736, 768])
        #-----------------------------处理两个模态一起的embedding-------------------------# 

        #我不想考虑mask的问题
        for i, blk in enumerate(self.transformer.blocks): #总共是12层哈
            # x, _attn = blk(x, mask=co_masks)
            x, _attn = blk(x)
            if torch.isnan(x).any():
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NAN in transformer")

        x = self.transformer.norm(x) 
        if torch.isnan(x).any():
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NAN in norm")
        ehr_feats, image_feats = ( #feats是 feature
            x[:, : ehr_embeds.shape[1]], #在这里，x[:, : text_embeds.shape[1]]表示选择 x 的所有行，
                                            #列从开始到 text_embeds.shape[1]（即文本嵌入的长度）。
                                            # 所以，text_feats就是对应文本的特征。
            x[:, ehr_embeds.shape[1] :], #同样，x[:, text_embeds.shape[1] :]表示选择 x 的所有行，
                                         #列从 text_embeds.shape[1] 到结尾。
                                         # 所以，image_feats就是对应图像的特征。
        )
        cls_feats = self.pooler(x)
        if torch.isnan(cls_feats).any():
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NAN in cls_feats")       

        ret = {
            "ehr_feats": ehr_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "ehr_labels": ehr_label,
            # "text_ids": text_ids,
            # "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        #这是我们的数据集哈，根据不同的数据集选择不同的下游任务
        if "mimic" in self.current_tasks:
            ret.update(objectives.compute_mimic(self, batch))


        return ret


    def training_step(self, batch, batch_idx):#定义训练过程每一步
        vilt_utils.set_task(self) #训练过程要干啥取决于set_task
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k]) #叠加起来每一个loss
        vilt_utils.epoch_wrapup(self)

        return total_loss

    def training_epoch_end(self, outs): #每个epoch结束后做的清理工作：如计算和打印训练周期的平均损失等
        # vilt_utils.epoch_wrapup(self)
        pass

    def validation_step(self, batch, batch_idx): #这个函数类似于training_step，但是它是用来定义在验证过程中的一步。
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs): #类似于training_epoch_end
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx): # 定义测试要干啥。和training_step很像
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config.load_path_test.split("/")[-1][:-5]

        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self): #这个函数是用来配置优化器的
        return vilt_utils.set_schedule(self)

    # def on_after_backward(self): #在每次反向传播的时候干点事情
        # last_layer_grad = self.mimic_classifier.get_last_layer().weight.grad.mean()
        # self.log('last_layer_grad',last_layer_grad)
        # with open('/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/output.txt', 'a') as f:
        #     f.write(str(last_layer_grad.item()))
        #     f.write('\n')

        #打印分类层的梯度看看
        # for name, param in self.mimic_classifier.named_parameters():
        #     if param.grad is not None:
        #         self.log(f'{name}_average_grad', param.grad.abs().mean())


    #在每次训练开始的时候干点事情,把重要的文件给复制到训练保存东西的文件夹中
    def on_train_start(self):
        files_to_copy = {
            'my_metrics.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/vilt/gadgets/my_metrics.py',
            'objectives.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/vilt/modules/objectives.py',
            'config.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/config.py',
            'Extract_data_1.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/Test/Extract_data_1.py',
            'vilt_module.py': '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/vilt/datamodules/vilt_module.py'
        }
        for file_name, source_path in files_to_copy.items():
            target_file_path = f'{self.trainer.logger.log_dir}/{file_name}'
            if not os.path.exists(target_file_path):
                shutil.copyfile(source_path, target_file_path)
