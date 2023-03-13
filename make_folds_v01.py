# pip install scikit-multilearn
import os
import sys
import random
import numpy as np 
import platform
'''
PATH = '/Users/dhanley/Documents/rsna-breast-cancer-detection'
os.chdir(f'{PATH}')
'''
import pandas as pd
import utils
from skmultilearn.model_selection import IterativeStratification
utils.set_pandas_display()

DEFAULT_RANDOM_SEED = 2021

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seedBasic(seed=0)

trnfile = 'datamount/train.csv'
trndf = pd.read_csv(trnfile)

# Pick out some fields to stratify on
aggdf = trndf['site_id patient_id age laterality implant machine_id cancer'.split()]
aggdf['age_bin'] = (aggdf.age.fillna( trndf.age.dropna().mean()) //10).astype(int)
aggdf = aggdf.drop('age', 1)

# drop dupes on image level to key on patient and laterality
aggdf = aggdf.drop_duplicates()

# Aggregate by mean cancer per patient
aggdf = aggdf.groupby('site_id patient_id age_bin implant machine_id'.split())['cancer'].mean().reset_index()

# We have one patient on two machine_ids; just drop one of the machine_ids
aggdf = aggdf.drop_duplicates('patient_id').reset_index(drop = True)

# straity the split based on site, implant, age, machine, cancer
k_fold = IterativeStratification(n_splits=5, order=2)
X = aggdf[['patient_id']]
y = aggdf.drop('patient_id',1).values
aggdf['fold'] = -1

for t, (_, test) in enumerate(k_fold.split(X, y)):
    aggdf.loc[test, 'fold'] = t

trndf['fold'] = aggdf.set_index('patient_id')['fold'].loc[trndf.patient_id].values
trndf.to_csv('datamount/train_folded_v01.csv', index = False)

