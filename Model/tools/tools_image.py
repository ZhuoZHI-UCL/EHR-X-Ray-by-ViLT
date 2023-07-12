import skimage, torchvision
import torch.nn as nn
import torch
import torchxrayvision as xrv
#--------------------------------------使用Medfuse的LSTM模型---------------------------------------------#
class XRayImageClassifier(nn.Module):
    def __init__(self):
        super(XRayImageClassifier, self).__init__()
        self.xray_encoder = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(1024, 1)) 


    def forward(self, x):
        x = self.xray_encoder.features(x)
        visual_feats = x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        preds = self.classifier(x)
        preds = torch.sigmoid(preds)
        return preds,visual_feats
