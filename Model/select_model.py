
from Model.ehr_only_model import ehr_only_model 
from Model.image_only_model import image_only_model
from Model.fuse_model import fuse_model
from Model.fuse_cxr_model import fuse_cxr_model
from Model.fuse_pretrained_model import fuse_pretrained_model
from Model.fuse_incontext_model import fuse_incontext_model     

from Config.select_config import mission
from Config.ehr_only_config import args_ehr_only
from Config.image_only_config import args_image_only
from Config.fuse_config import args_fuse
from Config.fuse_cxr_config import args_fuse_cxr
from Config.fuse_pretrained_config import args_fuse_pretrained
from Config.fuse_incontext_config import args_fuse_incontext

if mission == 'ehr_only':
    config=args_ehr_only()
    model=ehr_only_model(config)
elif mission == 'image_only':
    config=args_image_only()
    model = image_only_model(config)
elif mission == 'fuse':
    config=args_fuse()
    model = fuse_model(config)
elif mission == 'fuse_cxr':
    config=args_fuse_cxr()
    model = fuse_cxr_model(config)
elif mission == 'fuse_pretrained':
    config=args_fuse_pretrained()
    model = fuse_pretrained_model(config)
elif mission == 'fuse_incontext':
    config=args_fuse_incontext()
    model = fuse_incontext_model(config)