import pytorch_lightning as pl
import os
from Model.select_model import model
from Dataloader.datamodule import datamodule
from Config.select_config import args
import socket
import torch
import sys
if __name__ == '__main__':
    print('All started')
    #----------------------------------环境参数设置----------------------------------#
    _config=args
    os.environ["CUDA_VISIBLE_DEVICES"] = _config.gpu_id
    os.makedirs(_config.log_dir, exist_ok=True)
    pl.seed_everything(_config.seed)

    if _config.save_best:
        save_top_k = 1
    else:
        save_top_k = 0
    #训练时候我们让它时刻输出设备名称
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=save_top_k,
        verbose=True,
        # monitor="val_auroc/val_all_auroc",
        monitor="val_auroc/val_final_matrix",
        mode="max",
        save_last=True, 
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
    #--------------------------------打印环境设备设置--------------------------------#
    if _config.test_only:
        print('we are Testing the model')
    else:
        print('we are Training the model')
    print('we are running on Mission: {}'.format(_config.mission))
    print('we are running on Task: {}'.format(_config.task))
    print('we are using the dataset: {}'.format(_config.data_pairs))
    print('we are running on GPU No.{}'.format(_config.gpu_id))
    print(socket.gethostname())
    #----------------------------------数据与模型----------------------------------#
    Data = datamodule(_config)
    Model = model
    #打印样本的数量
    train_samples = len(Data.train_dataloader().dataset)
    print(f'Training samples: {len(Data.train_dataloader().dataset)}')
    print(f'Validation samples: {len(Data.val_dataloader().dataset)}')
    
    #----------------------------------训练与测试----------------------------------# 
    if _config.data_pairs == 'radiology':
        num_sanity_val_steps=0
    else:
        num_sanity_val_steps=0
    trainer = pl.Trainer(
        num_sanity_val_steps=num_sanity_val_steps,
        gpus=_config.num_gpus,
        num_nodes=1,
        precision=_config.precision,
        accelerator="cuda",
        benchmark=True,
        deterministic=True,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger( _config.log_dir,name=f'_mission_{_config.mission}_task_{_config.task}',),
        max_epochs=_config.max_epoch,
        max_steps=_config.max_steps,
        accumulate_grad_batches=_config.batch_size // (_config.per_gpu_batchsize * _config.num_gpus),
        # accumulate_grad_batches=2,
        log_every_n_steps=_config.log_every_n_steps, 
        resume_from_checkpoint=_config.resume_from,
        fast_dev_run=_config.fast_dev_run,
        val_check_interval=_config.val_check_interval,
    )

    if not _config.test_only:
        # try:
        #     trainer.fit(Model, datamodule=Data)
        # except Exception as e:
        #     with open(f"Bin/{socket.gethostname()} 0 {os.environ['STY']}.txt", 'w') as f:
        #         f.write('fuck')
            # sys.exit(1)
        # Model = model.load_from_checkpoint(checkpoint_path = _config.load_path , strict= False)
        trainer.fit(Model, datamodule=Data)
    else:
        model = Model.load_from_checkpoint(_config.load_path_test)
        trainer.test(model, datamodule=Data)