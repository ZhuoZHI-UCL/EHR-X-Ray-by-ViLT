#这个是旧的，这里我们考虑不加载预训练权重
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils

#如果我们不用文本的话，需要自己写一个时间序列的嵌入
#位置嵌入
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

#EHR数据嵌入
class EHREmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim, max_len=5000):
        super(EHREmbedding, self).__init__()
        self.feature_embedding = nn.Linear(feature_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        x = self.feature_embedding(x)
        # x: (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1) #这行代码是将输入张量 x 的第0维（通常对应批次大小）和第1维（通常对应序列长度）进行交换。
                              #这是因为在很多情况下，Transformer的实现需要输入的数据的形状为 (seq_len, batch_size, feature_dim)，
                              # 而不是 (batch_size, seq_len, feature_dim)。这样做的一个主要原因是可以方便地处理不同长度的序列。
        # x: (seq_len, batch_size, embed_dim)
        x = self.positional_encoding(x)
        # x: (seq_len, batch_size, embed_dim)
        return x





class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        '''
        #bert模型的config
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)
        '''
        
        

        #是否加载预训练权重
        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])#transformer的池化层，用于将 Transformer 编码器的输出特征进行池化，
                                                       #得到一个固定长度的特征向量，用于后续的分类或回归任务
        self.pooler.apply(objectives.init_weights)#一个初始化函数，用于初始化模型参数的权重和偏置，以便模型可以更快地收敛。
        

        #代理任务 mlm,itm,mpp
        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)#head模块的作用是从 Transformer 编码器的输出特征中提取对 MLM 任务有用的信息，
                                                        # 然后计算预测结果并计算损失。具体来说，它包含一个线性变换和一个 softmax 激活函数，
                                                        # 用于将 Transformer 编码器的输出特征映射到词汇表大小的输出空间
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (##还是加载预训练权重
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]#state_dict是ckpt这个权重的一个键，它是一个字典，用于存储模型的权重和偏置等参数
            self.load_state_dict(state_dict, strict=False) #将给定的状态字典 state_dict 加载到当前模型中，并更新模型的权重和偏置等参数。

        hs = self.hparams.config["hidden_size"]#768 transformer的参数

        #不同的下游任务使用不同的loss----------------------------------------#
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)


    #----------------------------------------前向传播过程-----------------------------------------#
    def infer( #实际上infer()在后面的调用中只调用了 self.infer(batch) batch一个参数，其他的都是默认参数
        self,
        batch, #batch 就是forward函数中的x
        # mask_ehr=False,#mask_text用于决定是否对输入的文本进行掩码操作,用于mlm
        mask_image=False,#mask_image决定是否 随机掩盖输入图像中的一部分，然后让模型预测被掩盖的部分。
        image_token_type_idx=1, #区分这两种类型的输入
        image_embeds=None, #输入的图像的嵌入向量
        image_masks=None, #用来指示图像中被遮掩住的地方（1和0来区分，掩码矩阵
    ):  
        
        #它看起来是在用 image_token_type_idx 来选择具体要处理的图像。
        # 如果 image_token_type_idx 为 1，那么它会寻找 batch 中的 "image_0"；
        # 如果 image_token_type_idx 为 2，那么它会寻找 "image_1"，
        # 以此类推。如果找不到对应的图像，就会默认使用 "image"。
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        #处理mlm代理任务的
        
        # do_mlm = "_mlm" if mask_ehr else ""#判断是否进行mask language model任务，如果进行，则在后续的key中增加'_mlm'后缀。
        # text_ids = batch[f"ehr_ids{do_mlm}"]#这行代码是从输入的batch中获取文本的输入数据。如果do_mlm为"_mlm"，则获取的key就是"text_ids_mlm"，否则就是"text_ids"。
        # ehr_labels = batch[f"ehr_labels{do_mlm}"]#从输入的batch中获取文本的标签数据。如果do_mlm为"_mlm"，则获取的key就是"text_labels_mlm"，否则就是"text_labels"。
        # ehr_masks = batch[f"ehr_masks"]#这行代码是从输入的batch中获取EHR数据的mask。其中，mask通常是一个与输入相同大小的二维矩阵，用于指定哪些输入数据应被模型忽略。

        #----------------------------处理EHR的区域----------------------------#
        ehr_embeds =  EHREmbedding(self.hparams.config["ehr_feature_size"], 
                                            self.hparams.config["hidden_size"], 
                                            self.hparams.config["ehr_max_len"])
        

        #----------------------------处理图像的区域----------------------------#
        if image_embeds is None and image_masks is None:#这两个是设置成了none，所以只用执行第一个，就是直接
            img = batch[imgkey][0]                     #用vit的嵌入来嵌入图像
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )


        #-----------------------------处理两个模态一起的embedding-------------------------#
        ehr_embeds, image_embeds = ( #这一步是给两个嵌入的模态分别加上模态标识
                                    #token_type_embeddings接收一个和输入嵌入向量同样大小的全0或全1的tensor，
                                    # 然后生成一个同样大小的嵌入向量，这个向量在加到原始的输入嵌入向量上，
                                    # 就起到了区分不同类型输入的作用。
            ehr_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([ehr_embeds, image_embeds], dim=1)#co_embeds就包含了输入的所有信息，即文本和图像的嵌入信息。
        co_masks = torch.cat([ehr_embeds, image_masks], dim=1)#text_masks和image_masks也在维度1上进行合并，得到co_masks。
        #这个co_masks对应于co_embeds，用来指示co_embeds中哪些位置的数据是有效的，哪些位置的数据是被掩盖的或者是无效的。

        x = co_embeds
        #-----------------------------处理两个模态一起的embedding-------------------------# 


        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
