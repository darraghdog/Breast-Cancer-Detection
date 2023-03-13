import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, os.path as osp
import pandas as pd
import pydicom
from PIL import Image
import argparse
'''
os.chdir('/Users/dhanley/Documents/rsna-breast-cancer-detection')
'''
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm
import multiprocessing as mp
import dicomsdl
import torch
import torch.nn.functional as F


def load_dicomsdl(fn):
    path = DATA_FOLDER + fn
    img = np.asarray(dicomsdl.open(path).toPilImage())
    return img

def linear_window(data, center, width):
    lower, upper = center - width // 2, center + width // 2
    data.clip(lower, upper, out=data)
    return data 

def load_dicom_manual_window_dicomsdl(fn, bit16 = False):
    path = fn
    dcm = dicomsdl.open(path)
    data = dcm.pixelData()
    
    try:
        invert = getattr(dcm, "PhotometricInterpretation", None) == "MONOCHROME1"
    except:
        invert = False
        
    try:
        voi_func = getattr(dcm, "VOILUTFunction", "LINEAR")
        if voi_func is None:
            voi_func = 'LINEAR'
        voi_func = voi_func.strip().upper()
    except:
        voi_func = 'LINEAR'
    
    try:
        elem = dcm["WindowCenter"]
        center = float(elem[0]) if type(elem) ==list else float(elem)
    except:
        center = None
    try:
        elem = dcm["WindowWidth"]
        width = float(elem[0]) if type(elem) ==list else float(elem)
    except:
        width = None
#     bits = int(getattr(dcm, "BitsStored", 16))
    
    if (voi_func in ["LINEAR","SIGMOID"]) & (center is not None) & (width is not None):
        data = linear_window(data, center, width)

    data = (data - data.min()) / (data.max() - data.min())
    if invert:
        data = 1 - data
    if bit16:
        data = data * ((2**16)-1)
        return data.astype("uint16")
    data = data * ((2**8)-1)
    return data.astype("uint8")

def create_dir(d):
    if not osp.exists(d):
        os.makedirs(d)

class FilterImage(torch.nn.Module):
    def __init__(self, anchor_size = 1000, kernel_size = 13):
        super(FilterImage, self).__init__()
        self.anchor_size = anchor_size
        self.kernel_size = (kernel_size, kernel_size)
        self.kernel = self.get_binary_kernel2d(self.kernel_size)
        self.resize_args = {'mode':'bilinear', 'align_corners':True,'antialias':True}
        self.padding = [(k - 1) // 2 for k in self.kernel_size]

    def forward(self, x):
        
        # Downsize to put images on similar scale for kernel
        x1 = x.unsqueeze(0).unsqueeze(0).float()
        orig_size = torch.tensor(x.size()).cpu()
        downsize = max(1, max(orig_size).item()/self.anchor_size)
        x1 = torch.nn.functional.interpolate(x1, scale_factor = 1 / downsize, **self.resize_args)
        
        b, c, h, w = x1.shape
        
        # Apply minimum filter
        x1 = F.conv2d(x1.reshape(b * c, 1, h, w), self.kernel, padding=self.padding, stride=1)
        x1 = x1.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW
        x1 = x1.min(2)[0]
        
        # Apply maximum filter
        x1 = F.conv2d(x1.reshape(b * c, 1, h, w), self.kernel, padding=self.padding, stride=1)
        x1 = x1.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW
        x1 = x1.max(2)[0]
        
        # Now filter image
        x1 = x1[0,0]
        # Index of non zeros
        y_idx = (x1>0).sum(1).float()
        x_idx = (x1>0).sum(0).float()
        y_idx = y_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1) 
        x_idx = x_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        x_idx = torch.nn.functional.interpolate(x_idx, (x.shape[1],1), mode = 'bilinear')[0,0,:,0]
        y_idx = torch.nn.functional.interpolate(y_idx, (x.shape[0],1), mode = 'bilinear')[0,0,:,0]
        
        xout = x[y_idx>0][:,x_idx>0]
        
        return xout
    
    def get_binary_kernel2d(self, window_size):
        
        r"""Create a binary kernel to extract the patches.
    
        If the window size is HxW will create a (H*W)xHxW kernel.
        """
        window_range = window_size[0] * window_size[1]
        kernel = torch.zeros(window_range, window_range)
        for i in range(window_range):
            kernel[i, i] += 1.0
        return kernel.view(window_range, 1, window_size[0], window_size[1])
    
    def _compute_zero_padding(kernel_size):
        r"""Utility function that computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--datadir', type=str, default="datamount/")
arg('--bit16', type=int, default=0)
arg('--n_cores', type=int, default=16)
args = parser.parse_args()
bit16 = args.bit16 = args.bit16>0
print(f'Args : {args}')

DATA_DIR = args.datadir
df = pd.read_csv(osp.join(DATA_DIR, "train.csv"))


if bit16:
    SAVE_PNG_DIR = osp.join(DATA_DIR, "train_cropped_pngs_16bit_v05")
else:
    SAVE_PNG_DIR = osp.join(DATA_DIR, "train_cropped_pngs_8bit_v05")
create_dir(SAVE_PNG_DIR)

fns = []
for row_idx, row in tqdm(df.iterrows(), total=len(df)):
    fn_in = osp.join(DATA_DIR, "train_images", str(row.patient_id), str(row.image_id) + ".dcm")
    if os.path.isfile(fn_in):
        cropped_fname = f"{row.patient_id}_{row.laterality}_{row['view']}_{row.image_id}.png"
        fn_out = osp.join(SAVE_PNG_DIR, cropped_fname)
        if not os.path.isfile(fn_out):
            fns.append(f'{fn_in}||{fn_out}')

filt = FilterImage(anchor_size = 1024, kernel_size = 13)

fnames = fns[400]
def load_and_save(fnames):
    try:
        fn_in, fn_out = fnames.split('||')
        # img = load_and_crop_black_dicomsdl(fn_in)
        img = load_dicom_manual_window_dicomsdl(fn_in, bit16 = bit16)
        x = torch.from_numpy(img)
        x = filt(x)
        imgout = x.numpy()
        status = cv2.imwrite(fn_out, imgout)
        return status
    except:
        return False

'''
Image.fromarray(img[::10,::10])
Image.fromarray(imgout[::10,::10])
'''

with mp.Pool(min(args.n_cores,len(fns))) as p:
    success_rate = list(p.imap(load_and_save,tqdm(fns, total = len(fns))))

success_rate = [i for i in success_rate if i is not None]
print(f'Number processed {len(success_rate)} success rate {pd.Series(success_rate).mean():0.2f}')






