#这是最新的
#-----------------------------文档说明--------------------------------#
#这个文件我们直接提取pairedEHR和CXR数据并进行预处理
#参考/scratch/uceezzz/Project/Mis_mul/MedFuse-main/fusion_main.py

#-----------------------------导入引用区--------------------------------#
# from config import config
# args=config()
import numpy as np
import os, sys
# 获取当前文件的绝对路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 将当前文件的路径添加到sys.path
sys.path.append(current_path)

from preprocessing import Discretizer, Normalizer
from ehr_dataset import get_datasets
from cxr_dataset import get_cxr_datasets
from fusion import load_cxr_ehr
from pathlib import Path
import joblib
from tqdm import tqdm
from colorama import Fore, Style
from torch.utils.data import DataLoader
import torch

#-----------------------------DIY函数区--------------------------------#
def read_timeseries(args):
    path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)

def my_collate(batch):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([torch.zeros(3, 384, 384) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[2] for item in batch])
    targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [x, img, targets_ehr, targets_cxr, seq_length, pairs]#返回值

def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length
#-----------------------------顺序执行区--------------------------------#
def get_dataloader(args):
#下面这一块都是关于EHR数据提取的设置
    discretizer = Discretizer(timestep=float(args.timestep),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')
    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(args.timestep)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)


    print('Start to make dataset')
    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args) #已经预处理过了
    cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args) #进行预处理了，transform是在这里进行的
    print('Start to make dataloader')
    #这句话里面设置的ehr cxr 为none或者0或者1 会直接传递给最终的 my_collate函数, 在my_collate函数里面修改会直接传递给最后
    train_dl, val_dl, test_dl  = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)

    print('Congratulation! Dataloader loaded')
    return train_dl, val_dl, test_dl
if __name__=='__main__':
    print('All started')   
    import os, sys
    # 将当前文件的路径添加到sys.path
    sys.path.append(os.path.abspath('/scratch/uceezzz/Project/Mis_mul/Soluation_mis'))
    from Config.select_config import args
    print(args.data_pairs)
    train_dl, val_dl, test_dl= get_dataloader(args)

    #关于CXR_UNI数据集的说明:x, img, targets_ehr, targets_cxr, seq_length, pairs
    #x:设置成了1*10的全0数据
    #img: 3*384*384的图像
    #targets_ehr: 长度为1的标量
    #targets_cxr: 长度为14的向量
    #seq_length:  全为1
    #pairs: 全为True

    #综上来看，我们要进行如下操作：1.把ehr初始化成48*76的全0矩阵 2.把pairs设置成全为False

    # val_dl=joblib.load('/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/val_dl.pkl')
    print('all dataloader loaded')
    # batch_val = next(iter(val_dl))
    #现在我们发现 dataloader中每个sample的格式是
    #x, img, y_ehr, y_cxr, seq_lengths, pairs

    #y = self.get_gt(y_ehr, y_cxr)
    # x = torch.from_numpy(x).float()
    # x = Variable(x.to(self.device), requires_grad=False)
    
    # y = Variable(y.to(self.device), requires_grad=False)
    # img = img.to(self.device)

    '''
    #查看所有样本中最长的时间序列，和时间序列的均值
    max_value = -float('inf')  # 初始化为负无穷大
    total_value = 0.0 
    sample_count = 0.0
    for data in test_dl:
        _, _, _, _, third_element, _ = data  # 假设每个样本都包含6个元素
        third_element_tensor = torch.Tensor(third_element)

        max_value_batch = torch.max(third_element_tensor)  # 计算当前批次中第三个元素的最大值
        total_value += torch.sum(third_element_tensor).item()
        sample_count += third_element_tensor.numel()
        if max_value_batch > max_value:  # 如果当前批次的最大值大于已知最大值，则更新最大值
            max_value = max_value_batch

    mean_value = total_value / sample_count  # 计算平均值
    print(max_value) #最大值是 train: 2085  val 981 test 2392
    print(mean_value) #均值是 train  102    val 99  test 107

    '''
    # #查看每个样本中的时间序列长度并绘制直方图
    # all_length_train=[]
    # all_length_val=[]
    # all_length_test=[]

    # for data in train_dl:
    #     _, _, _, _, third_element, _ = data #这个数据是batch*1的
    #     all_length_train.append(third_element)
    # for data in val_dl:
    #     _, _, _, _, third_element, _ = data #这个数据是batch*1的
    #     all_length_val.append(third_element)
    # for data in test_dl:
    #     _, _, _, _, third_element, _ = data #这个数据是batch*1的
    #     all_length_test.append(third_element)
    # #绘制直方图
    # import matplotlib.pyplot as plt
    # plt.hist(all_length_train, bins=10, alpha=0.5)
    # plt.savefig('/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/histogram_train.png')
    # plt.hist(all_length_val, bins=10, alpha=0.5)
    # plt.savefig('/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/histogram_val.png')
    # plt.hist(all_length_test, bins=10, alpha=0.5)
    # plt.savefig('/scratch/uceezzz/Project/Mis_mul/Soluation_ZhuoZHI/save/histogram_test.png')

    # #统计一下每个dataset里面不匹配的数量
    # # 初始化计数器
    # count1_train=count2_train=count1_val=count2_val=count1_test=count2_test=0

    # # 遍历 DataLoader 的每个批次
    # for batch in train_dl:
    #     # 假设第3个数据是我们关心的布尔值
    #     bool_values = batch[5]
        
    #     # 统计 False 值的数量
    #     count1_train += len(batch[5])
    #     count2_train += sum(value == False for value in bool_values)
    # print(count1_train)
    # print(count2_train)


    # for batch in val_dl:
    #     # 假设第3个数据是我们关心的布尔值
    #     bool_values = batch[5]
        
    #     count1_val += len(batch[5])
    #     count2_val += sum(value == False for value in bool_values)
    # print(count1_val)
    # print(count2_val)

    # for batch in test_dl:
    #     # 假设第3个数据是我们关心的布尔值
    #     bool_values = batch[5]
        
    #     count1_test += len(batch[5])
    #     count2_test += sum(value == False for value in bool_values)
    # print(count1_test)
    # print(count2_test)


    ##统计一下每个dataset里面missed_cxr的数量
    # m=0
    # for batch in test_dl:
    #     flag = batch[5]
    #     for i in flag:
    #         if i == 'missed_cxr':
    #             m = m +1

    print('All finished')