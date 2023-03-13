from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import timm
from torch.distributions import Beta


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            if len(Y.shape) == 1:
                Y = coeffs * Y + (1 - coeffs) * Y[perm]
            else:
                Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        return X, Y

def flatten(l):
    return [item for sublist in l for item in sublist]


class Net(nn.Module):

    def __init__(self, cfg: Any):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = self.cfg.n_classes
        # cfg.backbone =  'efficientnet_b0'
        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=cfg.pretrained, 
                                          num_classes=0, 
                                          global_pool="", 
                                          in_chans=self.cfg.in_channels)
        if cfg.gradient_checkpointing:
            self.backbone.set_grad_checkpointing()
    
        if 'efficientnet' in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]['num_chs']
        
        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(backbone_out, self.n_classes)
        self.head1 = torch.nn.Linear(backbone_out, 1) # difficult_case
        self.head2 = torch.nn.Linear(backbone_out, 1) # biopsy
        self.head3 = torch.nn.Linear(backbone_out, 4) # density
        self.head4 = torch.nn.Linear(backbone_out, 3) # birads
        self.return_embeddings = cfg.return_embeddings
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_ce = nn.CrossEntropyLoss()

        self.fea_conv = nn.Sequential( \
            nn.Conv1d(backbone_out, backbone_out, kernel_size=2, padding=0),
            nn.ReLU())
            
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

    def forward(self, batch):

        x = batch['input']
        
        # Get id of sample index in batch
        ids = [[t]*len(i) for t,i in enumerate(x)]
        ids = torch.tensor(flatten(ids), device = x[0].device)
        x = torch.cat(x).unsqueeze(1)
        
        with torch.no_grad():
            x = self.backbone(x)
            x = self.global_pool(x)
            x = x[:,:,0,0]
        
        # 2-way combinations within breast of gap layers 
        ids2 = []
        xc = []
        for idd in torch.unique(ids):
            idx1 = idd==ids
            idx2 = torch.combinations(torch.arange(idx1.sum(), 
                                                  device = x[0].device))
            xc.append(x[idx1][idx2])
            ids2  += [idd.item()] * len(x[idx1][idx2])
        ids2 = torch.tensor(ids2, device = x[0].device)
        xc = torch.cat(xc).permute(0,2,1)
        
        # 1D CNN over two way combinations
        xc = self.fea_conv(xc)
        xc = xc.squeeze(-1)
        
        # Average combos over each breast
        def get_agg_logits(logits, ids2):
            logits = torch.stack([logits[ids2==i].mean() for i in torch.unique(ids2)])
            return logits.unsqueeze(1)
        
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = get_agg_logits(self.head(dropout(xc)), ids2)
                logits1 = get_agg_logits(self.head1(dropout(xc)), ids2) # difficult_case
                logits2 = get_agg_logits(self.head2(dropout(xc)), ids2) # biopsy
            else:
                logits += get_agg_logits(self.head(dropout(xc)), ids2)
                logits1 += get_agg_logits(self.head1(dropout(xc)), ids2) # difficult_case
                logits2 += get_agg_logits(self.head2(dropout(xc)), ids2) # biopsy

        logits /= len(self.dropouts)
        logits1 /= len(self.dropouts)
        logits2 /= len(self.dropouts)
        
        outputs = {}
        if self.cfg.mode!='test':
            
            loss0 = self.loss_fn(logits,batch["target"].float().unsqueeze(1))
            loss1 = self.loss_fn(logits1,batch["difficult_case"].float().unsqueeze(1))
            loss2 = self.loss_fn(logits2,batch["biopsy"].float().unsqueeze(1))
            auxwt = batch['aux_weights'][0]
            outputs['loss'] = loss0 * (1-sum(auxwt[:2])) + loss1 * auxwt[0]  + loss2 * auxwt[1]
        
            idx3 = batch["density"] > -0.5
            idx4 = batch["birads"] > -0.5
            if sum(idx3) > 4:
                logits3 = self.head3(xc) # density
                logits3 = torch.stack([logits3[ids2==i].mean(0) for i in torch.unique(ids2)[idx3]])
                loss3 = self.loss_ce(logits3, batch["density"][idx3])
                outputs['loss_density'] = loss3
                outputs['loss'] += loss3 * auxwt[2]
            if sum(idx4) > 4:
                logits4 = self.head4(xc) # biopsy
                logits4 = torch.stack([logits4[ids2==i].mean(0) for i in torch.unique(ids2)[idx4]])
                loss4 = self.loss_ce(logits4,batch["birads"][idx4].long())
                outputs['loss_birads'] = loss4
                outputs['loss'] += loss4 * auxwt[3]
            
            outputs['loss_cancer'] = loss0 
            outputs['loss_difficult_case'] = loss1
            outputs['loss_biopsy'] = loss2
        
        if not self.training:
            outputs["logits"] = logits
            if self.return_embeddings:
                outputs["embedding"] = x
        return outputs
