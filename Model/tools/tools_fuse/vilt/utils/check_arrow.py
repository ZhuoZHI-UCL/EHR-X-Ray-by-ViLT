#这个文件用于查看我们转化完成的arrow文件
import pyarrow as pa
import pandas as pd

table = pa.ipc.open_file('/scratch/uceezzz/Project/Mis_mul/missing_aware_prompts-main/datasets/Food101/food101_val.arrow').read_all()
df = table.to_pandas()
print(df.head())