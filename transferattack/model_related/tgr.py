from functools import partial

import torch

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class TGR(MIFGSM):
    """
    NOTE:
        The Code Only Support Batchsize=1.
    """
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, **kwargs)
        self.model_name = model_name
        self.model      = self.model[1]
        self._register_model()
        self.model      = wrap_model(self.model.eval().cuda())
        
    def _register_model(self):
        def attn_tgr(module, grad_in, grad_out, gamma):
            mask        = torch.ones_like(grad_in[0]) * gamma
            out_grad    = mask * grad_in[0][:]
            if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224", "pit_b_224", "pit_ti_224", "visformer_small"]:
                B, C, H, W      = grad_in[0].shape
                out_grad_flat   = out_grad.view(B, C, H * W)
                out_grad_batch  = out_grad_flat[0]
                max_all         = torch.argmax(out_grad_batch, dim=1)
                max_all_H       = max_all // H
                max_all_W       = max_all % H
                min_all         = torch.argmin(out_grad_batch, dim=1)
                min_all_H       = min_all // H
                min_all_W       = min_all % H

                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, indices, max_all_H, :] = 0.0
                out_grad[:, indices, :, max_all_W] = 0.0
                out_grad[:, indices, min_all_H, :] = 0.0
                out_grad[:, indices, :, min_all_W] = 0.0

            if self.model_name in ["cait_s24_224"]:
                B, H, W, C = grad_in[0].shape
                out_grad_flat = out_grad.view(B, H * W, C)
                out_grad_batch = out_grad_flat[0]
                max_all = torch.argmax(out_grad_batch, dim=0)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = torch.argmin(out_grad_batch, dim=0)
                min_all_H = min_all // H
                min_all_W = min_all % H
                
                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, max_all_H, :, indices] = 0.0
                out_grad[:, :, max_all_W, indices] = 0.0
                out_grad[:, min_all_H, :, indices] = 0.0
                out_grad[:, :, min_all_W, indices] = 0.0
            return (out_grad, )
        
        def attn_cait_tgr(module, grad_in, grad_out, gamma):
            mask        = torch.ones_like(grad_in[0]) * gamma
            out_grad    = mask * grad_in[0][:]

            B, H, W, C = grad_in[0].shape
            out_grad_batch = out_grad[0, :, 0, :]
            
            max_all = torch.argmax(out_grad_batch, dim=0)
            min_all = torch.argmin(out_grad_batch, dim=0)
            
            indices = torch.arange(C, device=out_grad.device)
            out_grad[:, max_all, :, indices] = 0.0
            out_grad[:, min_all, :, indices] = 0.0
            # print(out_grad)
            return (out_grad, )
        
        def q_tgr(module, grad_in, grad_out, gamma):
            mask        = torch.ones_like(grad_in[0]) * gamma
            out_grad    = mask * grad_in[0][:]
            out_grad[:] = 0.0

            return (out_grad, grad_in[1], grad_in[2])
        
        def v_tgr(module, grad_in, grad_out, gamma):
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:                
                grad_in = list(grad_in)                  
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)

            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]   

            if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224", "pit_b_224", "pit_ti_224", "cait_s24_224"]:
                C               = grad_in[0].shape[2]
                out_grad_batch  = out_grad[0]
                max_all         = torch.argmax(out_grad_batch, dim=0)
                min_all         = torch.argmax(out_grad_batch, dim=0)
                
                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, max_all, indices] = 0.0
                out_grad[:, min_all, indices] = 0.0
            
            if self.model_name in ["visformer_small"]:
                B, C, H, W = grad_in[0].shape
                out_grad_flat = out_grad.view(B, C, H * W)
                out_grad_batch = out_grad_flat[0]
                
                max_all = torch.argmax(out_grad_batch, dim=1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = torch.argmin(out_grad_batch, dim=1)
                min_all_H = min_all // H
                min_all_W = min_all % H
                
                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, indices, max_all_H, max_all_W] = 0.0
                out_grad[:, indices, min_all_H, min_all_W] = 0.0        

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad, )
                else:
                    return_dics = return_dics + (grad_in[i], )              
            return return_dics
            
        def mlp_tgr(module, grad_in, grad_out, gamma):
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:
                grad_in = list(grad_in)
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)

            mask        = torch.ones_like(grad_in[0]) * gamma
            out_grad    = mask * grad_in[0][:]

            if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224" "pit_b_224", "pit_ti_224", "cait_s24_224"]:
                C               = grad_in[0].shape[2]
                out_grad_batch  = out_grad[0]
                max_all         = torch.argmax(out_grad_batch, dim=0)
                min_all         = torch.argmax(out_grad_batch, dim=0)
                
                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, max_all, indices] = 0.0
                out_grad[:, min_all, indices] = 0.0
            
            if self.model_name in ["visformer_small"]:
                B, C, H, W = grad_in[0].shape
                out_grad_flat = out_grad.view(B, C, H * W)
                out_grad_batch = out_grad_flat[0]
                
                max_all = torch.argmax(out_grad_batch, dim=1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = torch.argmin(out_grad_batch, dim=1)
                min_all_H = min_all // H
                min_all_W = min_all % H
                
                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, indices, max_all_H, max_all_W] = 0.0
                out_grad[:, indices, min_all_H, min_all_W] = 0.0

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad, )
                else:
                    return_dics = return_dics + (grad_in[i], )
            return return_dics
        
        attn_tgr_hook       = partial(attn_tgr, gamma=0.25)
        attn_cait_tgr_hook  = partial(attn_cait_tgr, gamma=0.25)
        q_tgr_hook          = partial(q_tgr, gamma=0.75)
        v_tgr_hook          = partial(v_tgr, gamma=0.75)
        mlp_tgr_hook        = partial(mlp_tgr, gamma=0.5)

        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.blocks[i].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name in ["pit_b_224"]:
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind  = block_ind
                elif block_ind >= 3 and block_ind < 9:
                    transformer_ind = 1
                    used_block_ind  = block_ind - 3
                elif block_ind >= 9 and block_ind < 13:
                    transformer_ind = 2
                    used_block_ind  = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name in ["pit_ti_224"]:
            for block_ind in range(12):
                if block_ind < 2:
                    transformer_ind = 0
                    used_block_ind  = block_ind
                elif block_ind >= 2 and block_ind < 8:
                    transformer_ind = 1
                    used_block_ind  = block_ind - 2
                elif block_ind >=8 and block_ind < 12:
                    transformer_ind = 2
                    used_block_ind  = block_ind - 8
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name in ["cait_s24_224"]:
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.blocks[block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.blocks[block_ind].mlp.register_backward_hook(mlp_tgr_hook)
                elif block_ind > 24 and block_ind < 26:
                    self.model.blocks_token_only[block_ind - 24].attn.attn_drop.register_backward_hook(attn_cait_tgr_hook)
                    self.model.blocks_token_only[block_ind - 24].attn.q.register_backward_hook(q_tgr_hook)
                    self.model.blocks_token_only[block_ind - 24].attn.k.register_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind - 24].attn.v.register_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind - 24].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name in ["visformer_small"]:
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.stage2[block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.stage2[block_ind].mlp.register_backward_hook(mlp_tgr_hook)
                elif block_ind >= 4 and block_ind < 8:
                    self.model.stage3[block_ind - 4].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.stage3[block_ind - 4].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.stage3[block_ind - 4].mlp.register_backward_hook(mlp_tgr_hook)
