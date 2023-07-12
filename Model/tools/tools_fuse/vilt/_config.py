
def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "mppd": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,        
    }
    ret.update(d)
    return ret
class config:
    #------------------------------重要的参数---------------------------------#
    gpu_id= "1,3"
    exp_name = "FUCK"
    per_gpu_batchsize = 48
    max_epoch = 20
    max_steps = 140
    data_root = "/scratch/uceezzz/Project/Mis_mul/missing_aware_independent/datasets/Food101/"
    num_gpus = sum(char.isdigit() for char in gpu_id)
    batch_size = num_gpus*per_gpu_batchsize*4
    load_path = "/scratch/uceezzz/Project/Mis_mul/missing_aware_independent/Weight_ViLT_"
    test_only = False
    num_workers = 1
    #------------------------------重要的参数---------------------------------#


    #带有 @ex.config 装饰器的函数中的局部变量会被
    #Sacred 搜集起来作为参数, 之后可以在任意函数中使用它们
    #参数的优先级 调用时传参 > Sacred 参数 > 默认参数
    #作为 Config Scope 的函数不能包含任何的 return 或者 yield 语句.
    #config中的参数会被外界输入的参数覆盖，比如：python config_demo.py print_config with a=6
    #所有与a有关系的参数都会被覆盖掉

    seed = 0
    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None

    # fix backbone model (ViLT) weights
    fix_model = True

    # missing modality config 设置缺失比例哈
    #看一下这个东西是怎么传入到模型里面去的
    missing_ratio = {'train': 0.7, 'val': 0.7, 'test': 0.7}
    missing_type = {'train': 'both', 'val': 'both', 'test': 'both'} # ['text', 'image', 'both'] in VL taskss
    both_ratio = 0.5   # missing both ratio
    missing_table_root = './datasets/missing_tables/'
    simulate_missing = False

    # missing_aware_prompts config
    prompt_type = 'input'
    prompt_length = 16
    learnt_p = True
    prompt_layers = [0,1,2,3,4,5]
    multi_layer_prompt = True    
        
    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384 #384
    max_image_len = -1
    patch_size = 32 #32
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1


    optim_type = "adamw"
    decay_power = 1
    end_lr = 0
    lr_mult = 1  


    get_recall_metric = False
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101    #101


    resume_from = None
    fast_dev_run = False
    finetune_first = False



    log_dir = "result"
    precision = 16
    exp_name = "finetune_food101"
    datasets = ["Food101"]
    loss_names = _loss_names({"food101": 1})
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
    #     optim_type = "adam"
    max_text_len = 512   
