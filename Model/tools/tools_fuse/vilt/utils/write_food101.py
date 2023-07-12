import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word

#这段代码的含义是指 首先去定位到一个图像，然后再去text.json里面去找对应的文本，这就说明其实text已经是提取好的
#然后split文件里面给出了划分的 train/test/val，实际上这个划分的是对应好的Food101数据集里面的划分的train和test

def make_arrow(root, dataset_root, single_plot=False, missing_type=None):
    image_root = os.path.join(root, 'images')#定位到image文件夹
    
    with open(f"{root}/class_idx.json", "r") as fp:
        FOOD_CLASS_DICT = json.load(fp)
        
    with open(f"{root}/text.json", "r") as fp:
        text_dir = json.load(fp)
        
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
        
    #举个例子：split是 "train"/"text"/"val",samples是 "frozen_yogurt_443.jpg",cls是"frozen_yogurt"，
    for split, samples in split_sets.items(): #split是指train/text/val某个类型 (有个问题阿 文件中是指明了val数据的)， samples是图像对应的具体名称
        split_type = 'train' if split != 'test' else 'test'
        data_list = []
        for sample in tqdm(samples):#tqdm(samples)提供一个进度条，以便您可以查看循环的执行进度。
            if sample not in text_dir:
                print("ignore no text data: ", sample)
                continue
            cls = sample[:sample.rindex('_')]#提取出samples中类别部分的字符串 比如sample叫red_velvet_cake_476.jpg，那么cls就是red_velvet_cake
            label = FOOD_CLASS_DICT[cls] #从class字典中提取出对应的cls，比如如果cls是 frozen_yogurt,那么就会返回字典里面定义的值：0
            image_path = os.path.join(image_root, split_type, cls, sample)#image_path是指向某张图像的具体目录

            with open(image_path, "rb") as fp:
                binary = fp.read()
                
            text = [text_dir[sample]]
            
            
            data = (binary, text, label, sample, split)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "text",
                "label",
                "image_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/food101_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        