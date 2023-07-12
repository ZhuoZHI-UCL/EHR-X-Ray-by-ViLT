import pytorch_lightning as pl
class args_image_only:
    mission = 'image_only'
    task='in-hospital-mortality'
    data_pairs='paired_ehr_cxr' # partial_ehr  paired_ehr_cxr
    learning_rate = 5e-5
    gpu_id= "3"#geneva
    per_gpu_batchsize =64 #72-55GB geneva 
    max_epoch = 60
    test_only = False
    num_workers=0

    val_check_interval = 1.0 #每多少个epoch验证一次
    log_every_n_steps=1 #每几步记录一次 tensorboard
    num_gpus = sum(char.isdigit() for char in gpu_id) 
    batch_size=num_gpus*per_gpu_batchsize*4
    max_steps =  int(max_epoch*18845/num_gpus/4/per_gpu_batchsize )
    exp_name = "FUCK"
    precision = 32
    ehr_max_len=48

    #--------------------------------------dataloader-------------------------------------------#
    ehr_data_dir= '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/dataset/Extracted_EHR'
    cxr_data_dir= '/scratch/uceezzz/Dataset/physionet.org/files/mimic-cxr-jpg-2.0.0.physionet.org'

    timestep=1.0
    normalizer_state=None
    resize = 224 #使用ViLT的时候应该还会再做一次resize
    crop = 224 #这两个参数后面可以调的
    pin_memory=True
    log_dir = "result"
    labels_set= 'mortality'
    data_ratio = 1.0
    #---------------------------------------训练过程---------------------------------------------- #
    test_exp_name = None
    seed = 1002


    draw_false_image = 0
    drop_rate = 0.1
    resume_from = None
    fast_dev_run = False
    lr_mult = 1  
    optim_type = "adam"
    decay_power = 1
    end_lr = 0
    get_recall_metric = False