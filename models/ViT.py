import torch
import torch.nn as nn
import torchvision

from models.vit_layers import TransformerEncoder
from models.data_stats import *


class ViT(nn.Module):
    def __init__(self, num_classes:int=10, dataset:str='cifar10', img_size:int=32, patch:int=8, dropout:float=0.,
                 num_layers:int=7, hidden:int=384, mlp_hidden:int=384, head:int=12, is_cls_token:bool=True, in_c:int=3):
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )
        
        # noramalization params
        self.normalize = torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        #self.mean = torch.tensor(STD_MEAN_DICT[dataset]['mean']).view(3, 1, 1)
        #self.std = torch.tensor(STD_MEAN_DICT[dataset]['std']).view(3, 1, 1)
        #self.mean_cuda = self.mean.to(torch.device("cuda"))
        #self.std_cuda = self.std.to(torch.device("cuda"))


    def forward(self, x):
        # normalize
        ##if x.is_cuda:
        ##    if self.mean_cuda is None:
        ##        self.mean_cuda = self.mean.to(x.device)
        ##        self.std_cuda = self.std.to(x.device)
        ##    out = (x - self.mean_cuda) / self.std_cuda
        ##else:
        ##    out = (x - self.mean) / self.std
        x = self.normalize(x)
      
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out