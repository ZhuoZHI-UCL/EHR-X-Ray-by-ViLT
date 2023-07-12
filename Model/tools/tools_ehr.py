import torch.nn as nn
import torch
#--------------------------------------去除ehr中的异常值---------------------------------------------#
def replace_outliers_with_mean(data, m=3):
    mean = data.mean(dim=1, keepdim=True)  # 按照样本的每个特征进行均值计算
    std = data.std(dim=1, keepdim=True)    # 按照样本的每个特征进行方差计算
    mask = (data - mean).abs() <= m * std
    data_out = data.clone()  # 创建一个数据的副本
    mean_expanded = mean.expand_as(data)
    data_out[~mask] = mean_expanded[~mask]  # 使用扩展后的均值替换异常值
    return data_out
#--------------------------------------使用Medfuse的LSTM模型---------------------------------------------#
class LSTM_Medfuse(nn.Module):
    def __init__(self, input_dim=76, num_classes=1, hidden_dim=256, batch_first=True, dropout=0.3, layers=2):
        super(LSTM_Medfuse, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        for layer in range(layers):
            setattr(self, f'layer{layer}', nn.LSTM(
                input_dim, hidden_dim,
                batch_first=batch_first,
                dropout = dropout
                )
            )
            input_dim = hidden_dim
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.feats_dim = hidden_dim
        self.dense_layer = nn.Linear(hidden_dim, num_classes)
        self.initialize_weights()
        # self.activation = torch.sigmoid
    def initialize_weights(self):
        for model in self.modules():

            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x, seq_lengths):
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        x = x.to(next(self.parameters()).device)
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht.squeeze()
        if self.do is not None:
            feats = self.do(feats)
        out = self.dense_layer(feats)
        scores = torch.sigmoid(out)
        return scores, feats
    
