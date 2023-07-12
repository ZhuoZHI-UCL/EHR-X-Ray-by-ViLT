import pytorch_lightning as pl
#这个模型是使用来自fuse_cxr的预训练权重，然后在partial_ehr_cxr上进行微调
class args_fuse_pretrained:
    mission = 'fuse_pretrained'
    task='in-hospital-mortality'
    data_pairs='partial_ehr_cxr' # partial_ehr  paired_ehr_cxr partial_ehr_cxr
    learning_rate = 2.5e-5
    gpu_id= "0"#geneva
    per_gpu_batchsize = 128 #72-55GB geneva 
    max_epoch = 40
    save_best = True #是否保存最好的模型
    num_workers= 1
    test_only = False
    weight_decay = 1e-1

    val_check_interval = 1.0 #每多少个epoch验证一次
    log_every_n_steps=1 #每几步记录一次 tensorboard
    num_gpus = sum(char.isdigit() for char in gpu_id) 
    batch_size=num_gpus*per_gpu_batchsize
    if data_pairs == 'partial_ehr_cxr':
        max_steps =  int(max_epoch*18688/batch_size)
    # batch_size = per_gpu_batchsize


    exp_name = "FUCK"
    precision = 32
    warmup_steps = 0
    decay_power = 1
    end_lr = 0

    #--------------------------------------模型的参数设置----------------------------------------#
    hidden_size= 768
    ehr_feature_size = 76
    ehr_max_len=48
    # max_image_len = ehr_max_len
    max_image_len = -1
    
    # missing_aware_prompts config
    prompt_type = 'input'
    prompt_length = 16
    learnt_p = True
    prompt_layers = [0,1,2,3,4,5]
    multi_layer_prompt = True    
    #--------------------------------------dataloader-------------------------------------------#
    # load_path = "/scratch/uceezzz/Project/Mis_mul/missing_aware_independent/Weight_ViLT_"
    load_path = '/scratch/uceezzz/Project/Mis_mul/Soluation_mis/result/_mission_fuse_cxr_task_in-hospital-mortality/version_32/checkpoints/epoch=13-step=17780.ckpt'
    vit = "vit_base_patch32_384"
    ehr_data_dir= '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/dataset/Extracted_EHR'
    cxr_data_dir= '/scratch/uceezzz/Dataset/physionet.org/files/mimic-cxr-jpg-2.0.0.physionet.org'
    load_path_test = ''

    timestep=1.0
    normalizer_state=None
    resize = 384 #使用ViLT的时候应该还会再做一次resize
    crop = 384 #这两个参数后面可以调的
    pin_memory=True
    log_dir = "result"
    labels_set= 'mortality'
    data_ratio = 1.0
    #---------------------------------------训练过程---------------------------------------------- #
    test_exp_name = None
    seed = 0


    draw_false_image = 0
    drop_rate = 0.1
    resume_from = None
    fast_dev_run = False
    lr_mult = 1  
    optim_type = "adamw"
    decay_power = 1
    end_lr = 0
    get_recall_metric = False