import sys, os, copy
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
import scipy as sp
from sklearn.metrics import log_loss, roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import f1_score


'''
import importlib
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

pp = importlib.import_module(cfg.post_process_pipeline)
pp_out = pp.post_process_pipeline(cfg, val_data, val_df)
'''

def pfbeta(labels, predictions, beta):
    #official implementation
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
#             cfp += 1 - prediction #bugfix
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.

# def pfbeta(labels, predictions, beta):
#     '''
    
#     Taken from https://www.kaggle.com/code/paarthbhatnagar/rsna-annotated-explanation-of-evaluation-metric
    
#     This function returns the probablistic fbeta-score.
#     Parameters
#         labels: 
#             - The ground truth labels.
#         predictions: 
#             - The probability of a certain datapoint belonging to a specific class.
#         beta:
#             - A parameter that determines the weight of recall in the combined score.
#     '''
#     y_true_count = 0
    
#     # probabilistic true positives
#     probabilistic_tp = 0
    
#     # probabilistic false positives
#     probabilistic_fp = 0
    
#     # Loop over every ground truth label.
#     for idx in range(len(labels)):
#         prediction = min(max(predictions[idx], 0), 1) # This line makes sure that the prediction probability is always between 0 and 1.
#         if (labels[idx]):
#             y_true_count += 1
#             probabilistic_tp += prediction
#             probabilistic_fp += 1 - prediction
#         else:
#             probabilistic_fp += prediction

#     beta_squared = beta * beta
    
#     # Probabilistic precision
#     c_precision = probabilistic_tp / (probabilistic_tp + probabilistic_fp)
    
#     # Probabilistic recall
#     c_recall = probabilistic_tp / y_true_count

#     # This part calculates the probabilistic Fbeta-Score
#     if (c_precision > 0 and c_recall > 0):
#         result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
#         return result
#     else:
#         return 0

def calc_metric(cfg, pp_out, val_df, pre="val"):

    pred_df = pp_out

    #aggregate by prediction_id , i.e. by patient_laterality
    val_df['prediction_id'] = val_df.apply(lambda x: f'{x.patient_id}_{x.laterality}', 1)
    val_df = val_df.groupby('prediction_id')[['cancer']].agg('mean') #doesnt matter if mean or max since all GT labels are consistent per predicition id
    
    # Sort both the same
    val_df = val_df.loc[pp_out.index]
    y = val_df['cancer'].values#.astype(np.float32)
    
    for pred_col in cfg.pred_columns:
        

    
        y_pred = pred_df[pred_col].values
        
        pF1_score = pfbeta(y, y_pred, 1)
        roc_score = roc_auc_score(y, y_pred)
        print(f"{pre} pF1_{pred_col}: {pF1_score:.6}")
        if hasattr(cfg, "neptune_run"):
            cfg.neptune_run[f"{pre}/pF1_{pred_col}/"].log(pF1_score, step=cfg.curr_step)
            
            cfg.neptune_run[f"{pre}/roc_{pred_col}/"].log(roc_score, step=cfg.curr_step)
#             print(f"{pre} roc_{pred_col}: {roc_score:.6}")
    
    return pF1_score


def calc_metric2(cfg, pp_out, val_df, pre="val"):

    pred_df, pred_df_img = pp_out
    val_df_img = val_df.copy()
    y_img = val_df_img['cancer'].values

    #aggregate by prediction_id , i.e. by patient_laterality
    if "prediction_id" not in val_df.columns:
        val_df['prediction_id'] = val_df.apply(lambda x: f'{x.patient_id}_{x.laterality}', 1)

    val_df = val_df.groupby('prediction_id')[['cancer']].agg('mean') #doesnt matter if mean or max since all GT labels are consistent per predicition id

    # Sort both the same
    val_df = val_df.loc[pred_df.index]
    y = val_df['cancer'].values#.astype(np.float32)

    for pred_col in cfg.pred_columns:
        if pred_col not in pred_df_img.columns:
            continue

        y_pred = pred_df_img[pred_col].values

        roc_score = roc_auc_score(y_img, y_pred)

        if cfg.neptune_run:
            cfg.neptune_run[f"{pre}/roc_{pred_col}_per_image/"].log(roc_score, step=cfg.curr_step)
        print(f"{pre} roc_{pred_col}: {roc_score:.6}")

    for pred_col in cfg.pred_columns:
        y_pred = pred_df[pred_col].values

        pF1_score = pfbeta(y, y_pred, 1)
        if "_pp" not in pred_col:
            roc_score = roc_auc_score(y, y_pred)

        if cfg.neptune_run:
            cfg.neptune_run[f"{pre}/pF1_{pred_col}/"].log(pF1_score, step=cfg.curr_step)
        print(f"{pre} pF1_{pred_col}: {pF1_score:.6}")

        if "_pp" not in pred_col:
            if cfg.neptune_run:
                cfg.neptune_run[f"{pre}/roc_{pred_col}/"].log(roc_score, step=cfg.curr_step)
            print(f"{pre} roc_{pred_col}: {roc_score:.6}")

    return pF1_score
