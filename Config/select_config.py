from Config.ehr_only_config import args_ehr_only 
from Config.image_only_config import args_image_only
from Config.fuse_config import args_fuse
from Config.fuse_cxr_config import args_fuse_cxr
from Config.fuse_pretrained_config import args_fuse_pretrained
from Config.fuse_incontext_config import args_fuse_incontext

mission = 'fuse' 
if mission == 'ehr_only':
    args = args_ehr_only()
elif mission == 'image_only':
    args = args_image_only()
elif mission == 'fuse':
    args = args_fuse() #使用partiacal_ehr_cxr直接训练
elif mission == 'fuse_cxr':
    args = args_fuse_cxr() #使用CXR_UNI预训练
elif mission == 'fuse_pretrained':
    args = args_fuse_pretrained() #使用CXR_UNI预训练，再用partiacal_ehr_cxr微调
elif mission == 'fuse_incontext':
    args = args_fuse_incontext() #使用ehr_cxr_partial数据集训练，使用incontext learning解决missing modality问题