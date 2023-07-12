import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from Model.tools.tools_fuse.vilt.modules.dist_utils import all_gather
# from vilt.modules.objectives import compute_irtr_recall
from Model.tools.tools_fuse.vilt.gadgets.my_metrics import Accuracy, VQAScore, Scalar, F1_Score, AUROC, Scalar2, check


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config.loss_names.items():
            if v < 1:
                continue
            if k == "mimic":#多分类的数据集
                setattr(pl_module, f"{split}_{k}_F1_scores", F1_Score())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
          
# def test_ablation(pl_module, loss_name, res):
#     test_ratio = pl_module.hparams.config['test_ratio']
#     exp_name = pl_module.hparams.config.test_exp_name
#     test_type = pl_module.hparams.config.test_type       
#     records = f'missing ratio: {test_ratio}, ' + res
#     record_file = f'./records/{loss_name}/{loss_name}_{exp_name}_on_missing_{test_type}'
#     with open(record_file, 'a+') as f:
#         f.write(records+'\n')
                
def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    for loss_name, v in pl_module.hparams.config.loss_names.items():
        if v < 1:
            continue

        value = 0

        if loss_name == "mimic":
            values = getattr(pl_module, f"{phase}_{loss_name}_F1_scores").compute()
            value = values[0]
            pl_module.log(f"{loss_name}/{phase}/auroc", values[0])
            pl_module.log(f"{loss_name}/{phase}/auprc", values[1])
            pl_module.log(f"{loss_name}/{phase}/F1_Weighted", values[2])
            pl_module.log(f"{loss_name}/{phase}/Hamming_loss", values[3])
            pl_module.log(f"{loss_name}/{phase}/accuracy", values[4])
            getattr(pl_module, f"{phase}_{loss_name}_F1_scores").reset()            
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            if pl_module.hparams.config.test_exp_name is not None:
                res = 'auroc: {0:.2f}, auprc: {1:.2f}, F1-Weighted: {2:.2f}, Hamming_loss: {3:.2f}, accuracy: {4:.2f}'.format(100*values[0], 100*values[1], 100*values[2], 100*values[3], 100*values[4])
                test_ablation(pl_module, loss_name, res) 
       
        the_metric += value

    pl_module.log(f"{phase}/the_metric_auroc", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config.loss_names.items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config.learning_rate
    wd = pl_module.hparams.config.weight_decay
    
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
    head_names = ["mimic_classifier"]
    lr_mult = pl_module.hparams.config.lr_mult
    end_lr = pl_module.hparams.config.end_lr
    decay_power = pl_module.hparams.config.decay_power
    optim_type = pl_module.hparams.config.optim_type

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps
    
    #线性下降的学习率
    # warmup_steps = pl_module.hparams.config.warmup_steps
    # if isinstance(pl_module.hparams.config.warmup_steps, float):
    #     warmup_steps = int(max_steps * warmup_steps)

    # if decay_power == "cosine":
    #     scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=warmup_steps,
    #         num_training_steps=max_steps,
    #     )
    # else:
    #     scheduler = get_polynomial_decay_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=warmup_steps,
    #         num_training_steps=max_steps,
    #         lr_end=end_lr,
    #         power=decay_power,
    #     )

    # sched = {"scheduler": scheduler, "interval": "step"}

    # return (
    #     [optimizer],
    #     [sched],
    # )

    from torch.optim.lr_scheduler import LambdaLR

    def lambda_lr(step: int):
        if step < max_steps/40:
            return 1
        else:
            return 0.1

    warmup_steps = pl_module.hparams.config.warmup_steps
    if isinstance(pl_module.hparams.config.warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    lambda_scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

    sched = {"scheduler": lambda_scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
    
