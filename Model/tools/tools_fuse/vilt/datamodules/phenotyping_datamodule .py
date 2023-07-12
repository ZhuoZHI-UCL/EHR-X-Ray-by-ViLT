from vilt.datasets import PHENOTYPINGDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class PHENOTYPINGDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PHENOTYPINGDataset

    @property
    def dataset_name(self):
        return "phenotyping"

    def setup(self, stage):
        super().setup(stage)

