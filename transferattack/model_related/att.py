import random
from functools import partial

from timm.models import create_model

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class ATT(MIFGSM):
    """
    NOTE:
        The Code Only Support Batchsize=1.
    """
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, lam=0.01, targeted=False, 
                 random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, **kwargs)
        self.image_size     = 224
        self.patch_size     = 16

        self.model_name     = model_name
        self.lam            = lam
        self.patch_index    = self.Patch_index(self.patch_size) # (1, 196, 256) 196个patch 每个patch包括256个像素
        self.im_fea         = None
        self.im_grad        = None
        
        self.model = self.model[1]
        self._register_model()
        self.model = wrap_model(self.model.eval().cuda())

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data    = data.clone().detach().to(self.device)
        label   = label.clone().detach().to(self.device)

        output  = self.model(data)
        # output.backward(torch.ones_like(output))
        output[:, label].sum().backward()
        resize  = transforms.Resize((224, 224))
        
        if self.model_name in ["vit_base_patch16_224"]:
            GF  = (self.im_fea[0][1:] * self.im_grad[0][1:]).sum(-1)
            GF  = resize(GF.reshape(1, 14, 14))
        elif self.model_name in ["deit_base_distilled_patch16_224"]:
            GF  = (self.im_fea[0][2:] * self.im_grad[0][2:]).sum(-1) 
            GF  = resize(GF.reshape(1, 14, 14))
        elif self.model_name in ["pit_b_224"]:
            GF  = (self.im_fea[0][1:] * self.im_grad[0][1:]).sum(-1)
            GF  = resize(GF.reshape(1, 8, 8))
        elif self.model_name in ["pit_ti_224"]:
            GF  = (self.im_fea[0][1:] * self.im_grad[0][1:]).sum(-1)
            GF  = resize(GF.reshape(1, 7, 7))  
        elif self.model_name in ["cait_s24_224"]:
            GF  = (self.im_fea[0] * self.im_grad).sum(-1)
            GF  = resize(GF.reshape(1, 14, 14))
        elif self.model_name in ["visformer_small"]:
            GF  = (self.im_fea[0] * self.im_grad[0]).sum(0)
            GF  = resize(GF.unsqueeze(0))

        GF_patch_t      = self.norm_patch(GF, self.patch_index, self.patch_size, self.scale, self.offset)
        GF_patch_start  = torch.ones_like(GF_patch_t).cuda() * 0.99
        GF_offset       = (GF_patch_start - GF_patch_t) / self.epoch

        delta       = self.init_delta(data)
        momentum    = 0
        
        for i in range(self.epoch):
            self.var_A      = 0
            self.var_qkv    = 0
            self.var_mlp    = 0
            if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
                self.back_attn  = 11
            elif self.model_name in ["pit_b_224"]:
                self.back_attn  = 12
            elif self.model_name in ["pit_ti_224"]:
                self.back_attn  = 11
            elif self.model_name in ["cait_s24_224"]:
                self.back_attn  = 24
            elif self.model_name in ["visformer_small"]:
                self.back_attn  = 7

            torch.manual_seed(i)
            random_patch    = torch.rand(14, 14).repeat_interleave(16).reshape(14, 14 * 16).repeat(1, 16).reshape(224, 224).cuda()
            GF_patch        = torch.where(torch.as_tensor(random_patch > (GF_patch_start - GF_offset * (i + 1))), 0., 1.).cuda()
            logits          = self.get_logits(self.transform(data + delta * GF_patch.detach()))
            loss            = self.get_loss(logits, label)
            grad            = self.get_grad(loss, delta)
            momentum        = self.get_momentum(grad, momentum)
            delta           = self.update_delta(delta, data, momentum, self.alpha)
        
        return delta.detach()

    def tr_01_pc(self, num, length):
        rate_l  = num
        tensor  = torch.cat((torch.ones(rate_l), torch.zeros(length - rate_l)))
        return tensor
    
    def _register_model(self):
        self.var_A      = 0
        self.var_qkv    = 0
        self.var_mlp    = 0
        self.gamma      = 0.5
        
        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            self.back_attn      = 11
            self.trunc_layers   = self.tr_01_pc(10, 12)
            self.weaken_factor  = [0.45, 0.7, 0.65]
            self.scale          = 0.4
            self.offset         = 0.4
        elif self.model_name in ["pit_b_224"]:
            self.back_attn      = 12
            self.trunc_layers   = self.tr_01_pc(9, 13)
            self.weaken_factor  = [0.25, 0.6, 0.65]
            self.scale          = 0.3
            self.offset         = 0.45
        elif self.model_name in ["pit_ti_224"]:
            self.back_attn      = 11
            self.trunc_layers   = self.tr_01_pc(9, 12)
            self.weaken_factor  = [0.25, 0.6, 0.65]
            self.scale          = 0.3
            self.offset         = 0.45
        elif self.model_name in ["cait_s24_224"]:
            self.back_attn      = 24
            self.trunc_layers   = self.tr_01_pc(4, 25)
            self.weaken_factor  = [0.3, 1., 0.6]
            self.scale          = 0.35
            self.offset         = 0.4
        elif self.model_name in ["visformer_small"]:
            self.back_attn      = 7
            self.trunc_layers   = self.tr_01_pc(8, 8)
            self.weaken_factor  = [0.4, 0.8, 0.3]
            self.scale          = 0.15
            self.offset         = 0.25

        def attn_att(module, grad_in, grad_out):
            mask        = torch.ones_like(grad_in[0]) * self.trunc_layers[self.back_attn] * self.weaken_factor[0]
            out_grad    = mask * grad_in[0][:]
            if self.var_A != 0:
                GPF = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_A))).clamp(0, 1)
            else:
                GPF = self.gamma
            
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
                out_grad[:, indices, max_all_H, :] *= GPF
                out_grad[:, indices, :, max_all_W] *= GPF
                out_grad[:, indices, min_all_H, :] *= GPF
                out_grad[:, indices, :, min_all_W] *= GPF

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
                out_grad[:, max_all_H, :, indices] *= GPF
                out_grad[:, :, max_all_W, indices] *= GPF
                out_grad[:, min_all_H, :, indices] *= GPF
                out_grad[:, :, min_all_W, indices] *= GPF

            self.var_A      = torch.var(out_grad)
            self.back_attn  -= 1
            return (out_grad,)
        
        def attn_cait_att(module, grad_in, grad_out):
            mask = torch.ones_like(grad_in[0]) * self.trunc_layers[self.back_attn] * self.weaken_factor[0]
            out_grad = mask * grad_in[0]
            
            if self.var_A != 0:
                GPF = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_A))).clamp(0, 1)
            else:
                GPF = self.gamma
            
            B, H, W, C = grad_in[0].shape
            out_grad_batch = out_grad[0, :, 0, :]
            
            max_all = torch.argmax(out_grad_batch, dim=0)
            min_all = torch.argmin(out_grad_batch, dim=0)
            
            indices = torch.arange(C, device=out_grad.device)
            out_grad[:, max_all, :, indices] *= GPF
            out_grad[:, min_all, :, indices] *= GPF

            self.var_A = torch.var(out_grad)
            self.back_attn -= 1
            
            return (out_grad,)

        def q_att(module, grad_in, grad_out):
            mask        = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad    = mask * grad_in[0][:]
            
            if self.var_qkv != 0:
                GPF = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv))).clamp(0, 1)
            else:
                GPF = self.gamma
            out_grad[:] *= GPF
            self.var_qkv = torch.var(out_grad)            
            return (out_grad, grad_in[1], grad_in[2])
        
        def v_att(module, grad_in, grad_out):
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:                
                grad_in = list(grad_in)                  
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0) 

            mask        = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad    = mask * grad_in[0][:]
            
            if self.var_qkv != 0:
                GPF = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv))).clamp(0, 1)
            else:
                GPF = self.gamma
            if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224", "pit_b_224", "pit_ti_224", "cait_s24_224"]:
                C               = grad_in[0].shape[2]
                out_grad_batch  = out_grad[0]
                max_all         = torch.argmax(out_grad_batch, dim=0)
                min_all         = torch.argmax(out_grad_batch, dim=0)
                
                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, max_all, indices]  *= GPF
                out_grad[:, min_all, indices]  *= GPF
            
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
                out_grad[:, indices, max_all_H, max_all_W] *= GPF
                out_grad[:, indices, min_all_H, min_all_W] *= GPF

            self.var_qkv = torch.var(out_grad)

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics
        
        def mlp_att(module, grad_in, grad_out):
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:
                grad_in = list(grad_in)
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)

            mask        = torch.ones_like(grad_in[0]) * self.weaken_factor[2]
            out_grad    = mask * grad_in[0][:]
            if self.var_mlp != 0:
                GPF = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_mlp))).clamp(0, 1)
            else:
                GPF = self.gamma

            if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224" "pit_b_224", "pit_ti_224", "cait_s24_224"]:
                C               = grad_in[0].shape[2]
                out_grad_batch  = out_grad[0]
                max_all         = torch.argmax(out_grad_batch, dim=0)
                min_all         = torch.argmax(out_grad_batch, dim=0)
                
                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, max_all, indices]  *= GPF
                out_grad[:, min_all, indices]  *= GPF
            
            if self.model_name in ["visformer_small"]:
                B, C, H, W = grad_in[0].shape
                out_grad_flat = out_grad.view(B, C, H * W)
                out_grad_batch = out_grad_flat[0]  # 取第一个批次
                
                max_all = torch.argmax(out_grad_batch, dim=1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = torch.argmin(out_grad_batch, dim=1)
                min_all_H = min_all // H
                min_all_W = min_all % H
                
                indices = torch.arange(C, device=out_grad.device)
                out_grad[:, indices, max_all_H, max_all_W] *= GPF
                out_grad[:, indices, min_all_H, min_all_W] *= GPF

            self.var_mlp = torch.var(out_grad)
            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)            
            return return_dics
        
        def get_im_fea(module, input, output):
            self.im_fea = output.clone()

        def get_im_grad(module, input, output):
            self.im_grad = output[0].clone()
            
        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            self.get_fea_hook   = self.model.blocks[10].register_forward_hook(get_im_fea)
            self.get_grad_hook  = self.model.blocks[10].register_backward_hook(get_im_grad)
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_att)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_att)
                self.model.blocks[i].mlp.register_backward_hook(mlp_att)
        elif self.model_name in ["pit_b_224"]:
            self.get_fea_hook   = self.model.transformers[2].blocks[2].register_forward_hook(get_im_fea)
            self.get_grad_hook  = self.model.transformers[2].blocks[2].register_backward_hook(get_im_grad)
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
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_att)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_att)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_att)
        elif self.model_name in ["pit_ti_224"]:
            self.get_fea_hook   = self.model.transformers[2].blocks[2].register_forward_hook(get_im_fea)
            self.get_grad_hook  = self.model.transformers[2].blocks[2].register_backward_hook(get_im_grad)
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
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_att)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_att)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_att)
        elif self.model_name in ["cait_s24_224"]:
            self.get_fea_hook   = self.model.blocks[23].register_forward_hook(get_im_fea)
            self.get_grad_hook  = self.model.blocks[23].register_backward_hook(get_im_grad)
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(attn_att)
                    self.model.blocks[block_ind].attn.qkv.register_backward_hook(v_att)
                    self.model.blocks[block_ind].mlp.register_backward_hook(mlp_att)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind - 24].attn.attn_drop.register_backward_hook(attn_cait_att)
                    self.model.blocks_token_only[block_ind - 24].attn.q.register_backward_hook(q_att)
                    self.model.blocks_token_only[block_ind - 24].attn.k.register_backward_hook(v_att)
                    self.model.blocks_token_only[block_ind - 24].attn.v.register_backward_hook(v_att)
                    self.model.blocks_token_only[block_ind - 24].mlp.register_backward_hook(mlp_att)
        elif self.model_name in ["visformer_small"]:
            self.get_fea_hook   = self.model.stage3[2].register_forward_hook(get_im_fea)
            self.get_grad_hook  = self.model.stage3[2].register_backward_hook(get_im_grad)
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(attn_att)
                    self.model.stage2[block_ind].attn.qkv.register_backward_hook(v_att)
                    self.model.stage2[block_ind].mlp.register_backward_hook(mlp_att)
                elif block_ind >= 4:
                    self.model.stage3[block_ind - 4].attn.attn_drop.register_backward_hook(attn_att)
                    self.model.stage3[block_ind - 4].attn.qkv.register_backward_hook(v_att)
                    self.model.stage3[block_ind - 4].mlp.register_backward_hook(mlp_att)

    def Patch_index(self, patch_size):
        """
        得到每个 patch 中像素点的索引位置   (整体是 从左向右 从上到下 的顺序)
        """
        filter_size = patch_size    # 16
        stride      = patch_size    # 16
        P           = np.floor((self.image_size - filter_size) / stride) + 1    # 14
        P           = P.astype(np.int32)    # 14
        Q           = P                     # 14
        index       = np.ones([P * Q, filter_size * filter_size], dtype=int)    # (196, 256)
        tmp_idx     = 0
        for q in range(Q):
            plus1   = q * stride * self.image_size  # q * 16 * 224
            for p in range(P):
                plus2   = p * stride                # p * 16
                index_  = np.array([], dtype=int)
                for i in range(filter_size):
                    plus    = i * self.image_size + plus1 + plus2   # i * 224 + q * 16 * 224 + p * 16
                    index_  = np.append(index_, np.arange(plus, plus + filter_size, dtype=int))
                index[tmp_idx]  = index_
                tmp_idx         += 1
        index   = torch.LongTensor(np.tile(index, (1, 1, 1))).cuda()        
        return index
    
    def norm_patch(self, GF, patch_index, patch_size, scale, offset):
        patch_area  = patch_size ** 2                   # 16 ** 2 = 256
        tmp         = torch.take(GF[0], patch_index[0]) # 取出对应 patch_index 中所有像素点的显著性分数
        norm_tmp    = torch.mean(tmp, dim=-1)           # 计算平均特征
        scale_norm  = scale * ((norm_tmp - norm_tmp.min()) / (norm_tmp.max() - norm_tmp.min())) + offset
        tmp_bi      = torch.as_tensor(scale_norm.repeat_interleave(patch_area)) * 1.0
        GF[0]       = GF[0].put_(patch_index[0], tmp_bi)
        return GF
