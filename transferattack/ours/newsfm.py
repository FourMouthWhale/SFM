from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class SFM(MIFGSM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, 
                 r=0.10, m=0.10, gamma=0.25,
                 targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, **kwargs)
        self.model_name     = model_name
        # global smoothing factor
        self.r              = r
        # base mixing factor
        self.m              = m
        # mixing growth coefficient
        self.gamma          = gamma

        self.model      = self.model[1]
        self._register_model()
        self.model      = wrap_model(self.model.eval().cuda())

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data    = data.clone().detach().to(self.device)
        label   = label.clone().detach().to(self.device)

        delta       = self.init_delta(data)
        momentum    = 0

        for _ in range(self.epoch):
            logits      = self.get_logits(self.transform(data + delta))
            loss        = self.get_loss(logits, label)
            grad        = self.get_grad(loss, delta)
            momentum    = self.get_momentum(grad, momentum)
            delta       = self.update_delta(delta, data, momentum, self.alpha)
        
        return delta.detach()
    
    def _register_model(self):
        def cross_sample_embedding_mix(module, input, output, ratio):
            """
            CEM: Cross-sample Embedding Mixing
            """
            batch_size = output.shape[0]
            if batch_size > 1:
                shuffle_idx = torch.randperm(batch_size, device=output.device)
                shuffled_output = output[shuffle_idx]
            else:
                shuffled_output = output
            
            mixed_output = ratio * shuffled_output.clone().detach() + (1 - ratio) * output
            
            return mixed_output
        
        def attn_ifs(module, input, output, mean_ratio=self.r):
            """
            IGR: Intra-sample Global Regularization
            """
            batch_size, num_heads, seq_length, _ = output.shape
            
            head_means = torch.mean(output, dim=(2, 3), keepdim=True)
            
            attn_map_mean = mean_ratio * head_means + (1 - mean_ratio) * output
            
            return attn_map_mean
        
        def qkv_ifs(module, input, output, mean_ratio=self.r):
            B, N, D = output.shape
            c_means = torch.mean(output, dim=1, keepdim=True)
            qkv_mean = mean_ratio * c_means + (1 - mean_ratio) * output
            return qkv_mean

        def q_ifs(module, input, output, mean_ratio=self.r):
            "cait"
            B, N = output.shape
            means = torch.mean(output, dim=1, keepdim=True)
            q_mean = mean_ratio * means + (1 - mean_ratio) * output
            return q_mean

        def kv_ifs(module, input, output, mean_ratio=self.r):
            "cait"
            B, N, D = output.shape
            c_means = torch.mean(output, dim=1, keepdim=True)
            kv_mean = mean_ratio * c_means + (1 - mean_ratio) * output
            return kv_mean
        
        def mlp_ifs(module, input, output, mean_ratio=self.r):
            B, N, D = output.shape
            c_means = torch.mean(output, dim=1, keepdim=True)
            mlp_mean = mean_ratio * c_means + (1 - mean_ratio) * output
            return mlp_mean

        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            total_layers = 12
            for block_ind in range(total_layers):
                # ifs
                self.model.blocks[block_ind].attn.qkv.register_forward_hook(qkv_ifs)
                self.model.blocks[block_ind].attn.attn_drop.register_forward_hook(attn_ifs)
                self.model.blocks[block_ind].mlp.register_forward_hook(mlp_ifs)
                # cem
                layer_ratio = block_ind / (total_layers - 1)
                m_ratio = self.m + layer_ratio * self.gamma
                self.model.blocks[block_ind].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                self.model.blocks[block_ind].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))