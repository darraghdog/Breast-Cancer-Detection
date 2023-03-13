import os
import sys
from importlib import import_module
import platform
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from albumentations.pytorch import ToTensorV2
import torch

sys.path.append("configs")
sys.path.append("augs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("postprocess")

from default_config import basic_cfg
import pandas as pd

cfg = basic_cfg
cfg.debug = True

# paths
# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"datamount/weights/{os.path.basename(__file__).split('.')[0]}"
cfg.data_dir = f"datamount/"
cfg.data_folder = cfg.data_dir + "train_cropped_pngs_8bit_v01/"
cfg.train_df = f'{cfg.data_dir}/train_folded_v01.csv'

# stages
cfg.test = False
cfg.test_data_folder = cfg.data_dir + "test_images/"
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1

#logging
cfg.neptune_project = "watercooled/rsna-screening"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"

#model
cfg.model = "mdl_dh_4k_agg2"
#cfg.backbone = 'seresnext50_32x4d'
cfg.backbone = 'tf_efficientnetv2_m.in1k'
# cfg.pretrained_weights = 'datamount/weights/cfg_ip_4c_aux14B_v2_m/fold-1/checkpoint_last_seed963175.pth'
# cfg.backbone = 'tf_efficientnet_b3.ns_jft_in1k'
'''
tf_efficientnet_b3.ns_jft_in1k	84.04
tf_efficientnet_b4.ns_jft_in1k	85.162
tf_efficientnet_b5.ns_jft_in1k	86.088
tf_efficientnet_b6.ns_jft_in1k	86.45
tf_efficientnet_b7.ns_jft_in1k	86.84
'''

cfg.pretrained = True
cfg.in_channels = 1
cfg.pool = 'gem'
cfg.gem_p_trainable = False
cfg.return_embeddings = False
cfg.mixup_beta =1

# OPTIMIZATION & SCHEDULE
cfg.fold = -1
cfg.epochs = 2

cfg.lr = 1e-4
cfg.optimizer = "Adam"
cfg.weight_decay = 1e-6
cfg.warmup = 0.3
cfg.batch_size = 16
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 8

# DATASET
cfg.dataset = "ds_dh_4j_aux8_agg1"
cfg.classes = ['cancer']
cfg.n_classes = len(cfg.classes)
# cfg.data_sample = 1000
cfg.normalization = "simple"
cfg.aux_weights = np.array([[0.2, 0.2, 0.2, 0.2] ] * cfg.epochs)
cfg.aux_weights[cfg.epochs//2:] /= 2
cfg.aux_weights_idx = ["difficult_case", "biopsy", "density", "birads"]
cfg.reload_train_loader = True
cfg.curr_epoch = 0
cfg.reload_train_loader = True
cfg.curr_epoch = 0
cfg.eval_steps = 450
cfg.mode='train_val'
cfg.gradient_checkpointing = False
cfg.find_unused_parameters = True

#EVAL
cfg.calc_metric = True
cfg.simple_eval = False
# augs & tta

# Postprocess
# augs & tta
cfg.pred_columns = ['cancer','cancer_pp','cancer_max','cancer_pp_max']

# Postprocess
cfg.post_process_pipeline = "pp_ch_1b"

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True
cfg.pretrained_weights_strict = False
cfg.pretrained_weights_strict = False

cfg.image_height = 1024
cfg.image_width = 1024

cfg.train_aug_pre = A.Compose([
        A.VerticalFlip(p=0.5 ),
        A.HorizontalFlip(p=0.5 ),
        A.Transpose(p=0.5 ),
        A.RandomRotate90(p=0.5 ),
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.2,
                           value = 1.0,
                           rotate_limit=0.15,
                           p=0.5,
                           border_mode = cv2.BORDER_CONSTANT),
        #A.OneOf([
        #    A.IAAEmboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
        #    A.IAAAdditiveGaussianNoise(loc=50, scale=(5, 12.75), per_channel=True, p=1.0),
        #], p=0.2),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
            A.Affine(shear = 10, p=1.),
            #A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.4),
        ])

cfg.train_aug = A.Compose([
        A.OneOf([
            A.RandomGridShuffle(grid=(4, 4), p=1.0),
            A.dropout.coarse_dropout.CoarseDropout(max_holes = 20, max_height=50, max_width=20,
                                       min_holes=5, fill_value=0, p = 1.),
        ], p=0.4),
        A.RandomCrop(always_apply=False, p=1.0, height=cfg.image_height, width=cfg.image_width),
        ])

cfg.val_aug = A.Compose([
        A.CenterCrop(always_apply=False, p=1.0, height=cfg.image_height, width=cfg.image_width),
        ])

cfg.train_aug_gpu = lambda x: x if np.random.choice(2)==1 else torch.rot90(x, np.random.choice([1,3]), [2, 3])
