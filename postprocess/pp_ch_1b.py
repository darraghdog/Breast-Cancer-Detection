import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

'''
import importlib, os, glob, copy, sys
os.chdir('/Users/dhanley/Documents/rsna-screening')
sys.path.append("configs")

cfg_name = 'cfg_dh_01A'
cfg = importlib.import_module(cfg_name)
cfg = copy.copy(cfg.cfg)
cfg.fold = 0
ds = importlib.import_module(cfg.dataset)
df = pd.read_csv('datamount/train_folded_v01.csv')
test_ds = ds.CustomDataset(df.query('fold==0'), cfg, cfg.val_aug, mode="valid")
val_df = test_ds.df

val_data_name = glob.glob(f'weights/{cfg_name}/fold0/val*')[0]
val_data = torch.load(val_data_name, map_location=torch.device('cpu'))
'''

def post_process_pipeline(cfg, val_data, val_df):
    
    cancer = torch.sigmoid(val_data['logits'].float()).cpu().numpy()[:,0]
#     cancer = (cancer > np.quantile(cancer,0.979)).astype(int)
    
    patient_id = val_df['patient_id'].values
    laterality = val_df['laterality'].values
    
    prediction_id = [f'{i}_{j}' for i,j in  zip(patient_id, laterality)]
    
    pp_out_raw = pd.DataFrame({'prediction_id': prediction_id, 'cancer': cancer})
    
    #aggregate by prediction_id , i.e. by patient_laterality
    pp_out = pp_out_raw.groupby('prediction_id')[['cancer']].agg('mean')
    pp_out['cancer_max'] = pp_out_raw.groupby('prediction_id')['cancer'].agg('max')
    
    #actual pp part
    pp_out['cancer_pp'] = (pp_out['cancer'].values > np.quantile(pp_out['cancer'].values,0.97935)).astype(int)
    pp_out['cancer_pp_max'] = (pp_out['cancer_max'].values > np.quantile(pp_out['cancer_max'].values,0.97935)).astype(int)
    # key_cols = 'patient_id laterality cancer'.split()
    # val_df[key_cols].drop_duplicates()
    
    return pp_out
