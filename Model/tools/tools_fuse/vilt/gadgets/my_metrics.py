import torch
from torchmetrics.functional import f1_score, auroc
from torchmetrics import Metric
from sklearn import metrics
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        if logits.size(-1)>1:
            preds = logits.argmax(dim=-1)
        else:
            preds = (torch.sigmoid(logits)>0.5).long()
            
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

class AUROC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
    
        if all_logits.size(-1)>1:
            all_logits = torch.softmax(all_logits, dim=1)
            AUROC = auroc(all_logits, all_targets, num_classes=2)
        else:
            all_logits = torch.sigmoid(all_logits)
            AUROC = auroc(all_logits, all_targets)
        
        return AUROC
    
class F1_Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:#将不同GPU上的张量给合并到一起
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
 




        if use_sigmoid:
            all_logits = torch.sigmoid(all_logits) #batchsize*25

        

        # #F1 score为评价指标
        # F1_Micro = f1_score(all_logits, all_targets,task='multilabel', average='micro',num_labels=25)
        # F1_Macro = f1_score(all_logits, all_targets,task='multilabel', average='macro',num_labels=25)   
        F1_Weighted = f1_score(all_logits, all_targets,task='multilabel', average='weighted',num_labels=25)#只考虑这个就行
        #这个F1是一个数，


        # auroc和auprc作为评价指标,我们希望的是输入是batchsize*25，输出是一个数
        y_true=all_targets.cpu().numpy()
        y_pred=all_logits.cpu().numpy()   


        #把输出写入csv看一下
        with open('/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/output.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([y_pred.mean()])

        # y_pred = (y_pred > 0.5).astype(int)
        #新增的指标 Hamming_loss
        # Hamming_loss=hamming_loss(y_true,y_pred)
        Hamming_loss=0
        auroc_each_class_all= []
        auprc_each_class_all=[]
        accuracy__each_class_all=[]
        for i in range(all_targets.shape[1]):
            if len(np.unique(y_true[:, i])) == 2:  # 只有在存在正样本和负样本时才计算
                # auroc_each_class = roc_auc_score(y_true[:, i], y_pred[:, i],average='weighted')
                # auroc_each_class_all.append(auroc_each_class * np.sum(y_true[:, i]))  # 对每个类别的AUC乘以其为1的样本数量
                auroc_each_class = roc_auc_score(y_true[:, i], y_pred[:, i])
                auroc_each_class_all.append(auroc_each_class)


                auprc_each_class = average_precision_score(y_true[:, i], y_pred[:, i],average='weighted')
                auprc_each_class_all.append(auprc_each_class * np.sum(y_true[:, i]))

                # accuracy_each_class=accuracy_score(y_true[:, i], y_pred[:, i])
                # accuracy__each_class_all.append(accuracy_each_class)

        # weighted_avg_auroc = np.sum(auroc_each_class_all) / np.sum(y_true)
        weighted_avg_auroc=np.mean(auroc_each_class_all)
        weighted_avg_auprc = np.sum(auprc_each_class_all) / np.sum(y_true)


        auroc= weighted_avg_auroc
        auprc= weighted_avg_auprc
        # accuracy= np.mean(accuracy__each_class_all)
        accuracy=0
        
        return (auroc, auprc, F1_Weighted,Hamming_loss,accuracy)
    
class check(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits).long()
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits.long()
            all_targets = self.targets.long()

        mislead = all_logits ^ all_targets
        accuracy = mislead.sum(dim=0)
        return accuracy
        
class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total    
    
class Scalar2(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar, num):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        
        self.scalar += scalar
        self.total += num

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total
