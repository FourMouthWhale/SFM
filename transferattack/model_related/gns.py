from functools import partial

import torch
from ..gradient.mifgsm import MIFGSM
from ..utils import *


class GNS(MIFGSM):
    """
    NOTE:
        The Code Only Support Batchsize=1.
    """
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, u=0.6, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, **kwargs)
        self.model_name = model_name
        self.u          = u
        self.model      = self.model[1]
        self._register_model()
        self.model      = wrap_model(self.model.eval().cuda())

    def _register_model(self):
        def attn_gns(module, grad_in, grad_out, gamma):
            mask        = torch.ones_like(grad_in[0]) * gamma
            out_grad    = mask * grad_in[0][:]
            return (out_grad, )

        def attn_cait_gns(module, grad_in, grad_out, gamma):
            mask        = torch.ones_like(grad_in[0]) * gamma
            out_grad    = mask * grad_in[0][:]
            return (out_grad, ) 
        
        def q_gns(module, grad_in, grad_out, gamma):
            mask        = torch.ones_like(grad_in[0]) * gamma
            out_grad    = mask * grad_in[0][:]
            return (out_grad, grad_in[1], grad_in[2])
        
        def v_gns(module, grad_in, grad_out):
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:                
                grad_in = list(grad_in)                  
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)

            if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224", "pit_b_224", "cait_s24_224"]:
                C           = grad_in[0].shape[2]
                c_mus       = torch.mean(torch.abs(grad_in[0]), dim=[0, 1])
                mu          = torch.mean(c_mus)
                std         = torch.std(c_mus)
                muustd      = mu + self.u * std
                c_factor    = c_mus > muustd
                c_temp      = torch.tanh(torch.abs((c_mus - mu) / std))
                c_factor    = c_factor.to(torch.float32)
                c_factor    = torch.where(c_factor == 0, c_temp, c_factor)
                grad_in[0] = grad_in[0] * c_factor.to(grad_in[0].device).view(1, 1, C)

            if self.model_name in ["visformer_small"]:
                B, C, H, W  = grad_in[0].shape
                c_mus       = torch.mean(torch.abs(grad_in[0]), dim=[0, 2, 3])
                mu          = torch.mean(c_mus)
                std         = torch.std(c_mus)
                muustd      = mu + self.u * std
                c_factor    = c_mus > muustd
                c_temp      = torch.tanh(torch.abs((c_mus - mu) / std))
                c_factor    = c_factor.to(torch.float32)
                c_factor    = torch.where(c_factor == 0, c_temp, c_factor)
                grad_in[0][:, :, :, :] = grad_in[0][:, :, :, :] * c_factor.to(grad_in[0].device).view(1, C, 1, 1)

            mask = torch.ones_like(grad_in[0])
            out_grad = mask * grad_in[0]

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad, )
                else:
                    return_dics = return_dics + (grad_in[i], )
            return return_dics
        
        def mlp_gns(module, grad_in, grad_out, gamma):
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:
                grad_in = list(grad_in)
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)

            mask        = torch.ones_like(grad_in[0]) * gamma
            out_grad    = mask * grad_in[0][:]
            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad, )
                else:
                    return_dics = return_dics + (grad_in[i], )
        
        attn_gns_hook       = partial(attn_gns, gamma=0.5)
        attn_cait_gns_hook  = partial(attn_cait_gns, gamma=0.5)
        v_gns_hook          = v_gns
        q_gns_hook          = partial(q_gns, gamma=0.5)
        mlp_gns_hook        = partial(mlp_gns, gamma=0.5)

        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_gns_hook)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_gns_hook)
                self.model.blocks[i].mlp.register_backward_hook(mlp_gns_hook)
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
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_gns_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_gns_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_gns_hook)
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
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_gns_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_gns_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_gns_hook)
        elif self.model_name in ["cait_s24_224"]:
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(attn_gns_hook)
                    self.model.blocks[block_ind].attn.qkv.register_backward_hook(v_gns_hook)
                    self.model.blocks[block_ind].mlp.register_backward_hook(mlp_gns_hook)
                elif block_ind > 24 and block_ind < 26:
                    self.model.blocks_token_only[block_ind - 24].attn.attn_drop.register_backward_hook(attn_cait_gns_hook)
                    self.model.blocks_token_only[block_ind - 24].attn.q.register_backward_hook(q_gns_hook)
                    self.model.blocks_token_only[block_ind - 24].attn.k.register_backward_hook(v_gns_hook)
                    self.model.blocks_token_only[block_ind - 24].attn.v.register_backward_hook(v_gns_hook)
                    self.model.blocks_token_only[block_ind - 24].mlp.register_backward_hook(mlp_gns_hook)
        elif self.model_name in ["visformer_small"]:
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(attn_gns_hook)
                    self.model.stage2[block_ind].attn.qkv.register_backward_hook(v_gns_hook)
                    self.model.stage2[block_ind].mlp.register_backward_hook(mlp_gns_hook)
                elif block_ind >= 4 and block_ind < 8:
                    self.model.stage3[block_ind - 4].attn.attn_drop.register_backward_hook(attn_gns_hook)
                    self.model.stage3[block_ind - 4].attn.qkv.register_backward_hook(v_gns_hook)
                    self.model.stage3[block_ind - 4].mlp.register_backward_hook(mlp_gns_hook)
