#这个文件我们使用partial_ehr_cxr数据集进行训练和测试
import pytorch_lightning as pl

class args_fuse:
    mission = 'fuse'
    task='phenotyping' # in-hospital-mortality phenotyping
    data_pairs='partial_ehr_cxr' # partial_ehr  paired_ehr_cxr partial_ehr_cxr  paired_ehr paired_cxr
    learning_rate = 3.75e-5
    per_gpu_batchsize = 192  
    max_epoch = 40
    num_workers=3
    gpu_id= "0,2,3"
    
    missing_prompt = True   #True False
    test_only = False       #True False
    preload_val = False     #True False

    save_best = True
    weight_decay = 5e-2
    val_check_interval = 1.0 #每多少个epoch验证一次
    log_every_n_steps=1 #每几步记录一次 tensorboard
    num_gpus = sum(char.isdigit() for char in gpu_id) 
    batch_size=num_gpus*per_gpu_batchsize*2
    reweight_loss = False 
    if task == 'in-hospital-mortality':
        labels_set= 'mortality'
        ehr_max_len=48      
        num_classes = 2 
        max_image_len = -1
        if data_pairs == 'partial_ehr_cxr' or data_pairs == 'partial_ehr':
            max_steps =  int(max_epoch*18845/(per_gpu_batchsize*num_gpus))
        elif data_pairs == 'paired_ehr_cxr' or data_pairs == 'paired_ehr' or data_pairs == 'paired_cxr':
            max_steps =  int(max_epoch*4885/(per_gpu_batchsize*num_gpus))

    elif task == 'phenotyping':
        labels_set= 'pheno'
        ehr_max_len= 96
        num_classes = 25
        max_image_len = 96

        if data_pairs == 'partial_ehr_cxr' or data_pairs == 'partial_ehr':
            max_steps =  int(max_epoch*42628/(per_gpu_batchsize*num_gpus))
        elif data_pairs == 'paired_ehr_cxr' or data_pairs == 'paired_ehr':
            max_steps =  int(max_epoch*7756/(per_gpu_batchsize*num_gpus))
    
    exp_name = "FUCK"
    precision = 32
    warmup_steps = 0
    decay_power = 1
    end_lr = 0

    #--------------------------------------模型的参数设置----------------------------------------#
    hidden_size= 768
    ehr_feature_size = 76

    
    # missing_aware_prompts config
    prompt_type = 'input'
    prompt_length = 16
    learnt_p = True
    prompt_layers = [0,1,2,3,4,5]
    multi_layer_prompt = True  
 
    #--------------------------------------dataloader-------------------------------------------#
    load_path = "/scratch/uceezzz/Project/Mis_mul/missing_aware_independent/Weight_ViLT_"
    vit = "vit_base_patch32_384"
    ehr_data_dir= '/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/dataset/Extracted_EHR'
    cxr_data_dir= '/scratch/uceezzz/Dataset/physionet.org/files/mimic-cxr-jpg-2.0.0.physionet.org'
    # load_path_test = '/scratch/uceezzz/Project/Mis_mul/Soluation_mis/result/_mission_fuse_task_in-hospital-mortality/version_102/checkpoints/epoch=77-step=11466.ckpt'
    load_path_test = '/scratch/uceezzz/Project/Mis_mul/Soluation_mis/result/_mission_fuse_task_in-hospital-mortality/version_212/checkpoints/epoch=44-step=3285.ckpt'
    timestep=1.0
    normalizer_state=None
    resize = 384 #使用ViLT的时候应该还会再做一次resize
    crop = 384 #这两个参数后面可以调的
    pin_memory=True
    log_dir = "result"

    data_ratio = 1.0
    #---------------------------------------训练过程---------------------------------------------- #
    test_exp_name = None
    seed = 1002


    draw_false_image = 0
    drop_rate = 0.1
    resume_from = None
    fast_dev_run = False
    lr_mult = 1  
    optim_type = "adamw"
    decay_power = 1
    end_lr = 0
    get_recall_metric = False