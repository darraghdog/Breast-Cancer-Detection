import platform
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def collate_fn(batch):
    # Remove error reads
    batch = [b for b in batch if b is not None]

    cols = 'input target difficult_case biopsy density birads aux_weights'.split()
    batchout = dict((k,torch.stack([b[k] for b in batch])) for k in cols)

    return batchout

tr_collate_fn = collate_fn
val_collate_fn = collate_fn

'''
df = pd.read_csv(cfg.train_df)

pd.crosstab(df.invasive, df.cancer)


aug = cfg.train_aug
cfg.val_aug
mode="valid"
class self:
    1
self = CustomDataset(df, cfg, aug = cfg.train_aug, mode = 'train')
self = CustomDataset(df, cfg, aug = cfg.val_aug, mode = 'valid')

batch = [self.__getitem__(i) for i in range(0, 8*50, 50)]
batch = tr_collate_fn(batch)
batch = batch_to_device(batch, 'cpu')
'''

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        self.df = self.df[self.df['image_id'].astype(str) != '1942326353'].copy()
        '''
        self.df = self.df.query('patient_id == 12988').query('laterality=="R"').reset_index(drop=True)
        '''
        self.mode = mode
        self.labels = self.df[self.cfg.classes].values
        self.df["fns"] = self.df['patient_id'].astype(str) + '_' + self.df['laterality'].astype(str) + '_' + self.df['view'].astype(str) + '_'+ self.df['image_id'].astype(str) + '.png'
        self.fns = self.df["fns"].astype(str).values
        self.aug = aug
        self.augpre = cfg.train_aug_pre
        
        self.df['BIRADS'] = self.df['BIRADS'].fillna(-1)
        self.df['density'] = self.df.density.map({'A':0,'B':1,'C':2,'D':3, np.nan:-1})
        
        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder
        # assert self.cfg.image_width == self.cfg.image_height
        idx = 100

    def __getitem__(self, idx):
        
        # for idx in range(300):
        
        samp = self.df.iloc[idx]
        difficult_case = int(samp.difficult_negative_case) + samp.cancer
        biopsy = samp.biopsy
        density = samp.density
        birads = samp.BIRADS
        
        label = self.labels[idx]
        img = self.load_one(idx)
        img = img[:,:,0]
        if max(img.shape) < self.cfg.image_width:
            hh, ww = img.shape
            scale = self.cfg.image_width * 1.125 / max(hh, ww)
            new_hh, new_ww = int(round(hh * scale)), int(round(ww * scale))
            img = Image.fromarray(img).resize((new_ww, new_hh), Image.Resampling.LANCZOS)
            img = np.array(img)
        
        # Pad, augment and resize large image
        padsize = abs(img.shape[0] - img.shape[1])
        if padsize > 2:
            img = self.pad_sides(img, padsize//5)
        
        if (self.augpre is not None) & (self.mode=='train'):
            img = self.augpre(image = img)['image']
        
        img = Image.fromarray(img)
        assert self.cfg.image_width == self.cfg.image_height
        if self.mode == 'train':
            new_size = int(self.cfg.image_width * np.random.uniform(1, 1.25))
        else:
            new_size = int(self.cfg.image_width * 1.125)
        img.thumbnail((new_size, new_size), Image.Resampling.LANCZOS)
        img = np.array(img)
        
        # No pad short side to square for smaller image. Randomly pad for train. 
        if (self.mode=='train'):
            dimgap = abs(img.shape[0] - img.shape[1])
            padsize = int(np.random.uniform(0, dimgap ))
            if padsize > 2:
                img = self.pad_sides(img, padsize)
            img = Image.fromarray(img)
            img = img.resize((new_size, new_size), Image.Resampling.LANCZOS)
            img = np.array(img)
        else:
            padsize = abs(img.shape[0] - img.shape[1])
            if padsize > 2:
                img = self.pad_sides(img, padsize)
        
        img = img[:,:,np.newaxis]
        if self.aug:
            img = self.augment(img)
        
        img = self.normalize_img(img)
        torch_img = torch.tensor(img).float().permute(2,0,1)
        aux_weights = self.cfg.aux_weights[self.cfg.curr_epoch]
        
        feature_dict = {
            "input": torch_img,
            "target": torch.tensor(label),
            "difficult_case": torch.tensor(difficult_case), 
            "biopsy": torch.tensor(biopsy), 
            "density": torch.tensor(density), 
            "birads": torch.tensor(birads),
            "aux_weights": torch.tensor(aux_weights),
        }
        return feature_dict

    def __len__(self):
        return len(self.fns)

    def load_one(self, idx):
        path = self.data_folder + self.fns[idx]
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            shape = img.shape
            if len(img.shape) == 2:
                img = img[:,:,None]
        except Exception as e:
            print(e)
        return img

    def augment(self, img):
        img = img.astype(np.float32)
        transformed = self.aug(image=img)
        trans_img = transformed["image"]
        return trans_img

    def normalize_img(self, img):

        if self.cfg.normalization == "image":
            img = (img - img.mean()) / (img.std() + 1e-4)
            img = img.clip(-20, 20)

        elif self.cfg.normalization == "simple":
            img = img / 255

        elif self.cfg.normalization == "min_max":
            img = img - np.min(img)
            img = img / np.max(img)

        return img
    
    def pad_sides(self, img, padsize):
        
        paddim = int(img.shape[0] > img.shape[1])
        if paddim==1:
            padimg = np.zeros((img.shape[0], padsize//2), dtype = np.uint8)
        else:
            padimg = np.zeros((padsize//2, img.shape[1]), dtype = np.uint8)
        img = np.concatenate((padimg, img, padimg), paddim)
        return img