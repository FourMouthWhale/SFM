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
        
        # def qkv_ifs(module, input, output, mean_ratio=self.r):
        #     B, N, D = output.shape
        #     c_means = torch.mean(output, dim=1, keepdim=True)
        #     qkv_mean = mean_ratio * c_means + (1 - mean_ratio) * output
        #     return qkv_mean
        
        def qkv_ifs(module, input, output, mean_ratio=self.r, process_parts=['q', 'k']):
            B, N, total_D = output.shape
            # 拆分QKV：总维度3*D，所以每个部分维度为D
            D = total_D // 3
            q, k, v = torch.split(output, D, dim=-1)
            
            # 定义处理逻辑：对指定部分进行均值融合，未指定部分保持原样
            def process_part(x):
                c_means = torch.mean(x, dim=1, keepdim=True)  # 沿序列维度N求均值
                return mean_ratio * c_means + (1 - mean_ratio) * x
            
            # 根据指定的部分进行处理
            processed_q = process_part(q) if 'q' in process_parts else q
            processed_k = process_part(k) if 'k' in process_parts else k
            processed_v = process_part(v) if 'v' in process_parts else v
            
            # 重新拼接处理后的QKV
            qkv_mean = torch.cat([processed_q, processed_k, processed_v], dim=-1)
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
                # # cem
                # layer_ratio = block_ind / (total_layers - 1)
                # m_ratio = self.m + layer_ratio * self.gamma
                # self.model.blocks[block_ind].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                # self.model.blocks[block_ind].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
        elif self.model_name in ["pit_ti_224"]:
            total_layers = 12
            for block_ind in range(total_layers):
                layer_ratio = block_ind / (total_layers - 1)
                m_ratio = self.m + layer_ratio * self.gamma
                if block_ind < 2:
                    transformer_ind = 0
                    used_block_ind  = block_ind
                elif block_ind >= 2 and block_ind < 8:
                    transformer_ind = 1
                    used_block_ind  = block_ind - 2
                elif block_ind >= 8 and block_ind < 12:
                    transformer_ind = 2
                    used_block_ind  = block_ind - 8
                # ifs
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_forward_hook(qkv_ifs)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_forward_hook(attn_ifs)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_forward_hook(mlp_ifs)
                # cem
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
        elif self.model_name in ["cait_s24_224"]:
            total_layers = 26
            for block_ind in range(26):
                layer_ratio = block_ind / (total_layers - 1)
                m_ratio = self.m + layer_ratio * self.gamma 
                if block_ind < 24:
                    # ifs
                    self.model.blocks[block_ind].attn.qkv.register_forward_hook(qkv_ifs)
                    self.model.blocks[block_ind].attn.attn_drop.register_forward_hook(attn_ifs)
                    self.model.blocks[block_ind].mlp.register_forward_hook(mlp_ifs)
                    # cem
                    self.model.blocks[block_ind].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                    self.model.blocks[block_ind].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                else:
                    # ifs
                    self.model.blocks_token_only[block_ind - 24].attn.q.register_forward_hook(q_ifs)
                    self.model.blocks_token_only[block_ind - 24].attn.k.register_forward_hook(kv_ifs)
                    self.model.blocks_token_only[block_ind - 24].attn.v.register_forward_hook(kv_ifs)
                    self.model.blocks_token_only[block_ind - 24].attn.attn_drop.register_forward_hook(attn_ifs)
                    self.model.blocks_token_only[block_ind - 24].mlp.register_forward_hook(mlp_ifs)
                    # cem
                    self.model.blocks_token_only[block_ind - 24].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                    self.model.blocks_token_only[block_ind - 24].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))



class IFS(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.1, gamma=0.25, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)
    
    def _register_model(self):
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

class IFS01(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.1, gamma=0.25, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)
    
    def _register_model(self):
        def attn_ifs(module, input, output, mean_ratio=self.r):
            """
            IGR: Intra-sample Global Regularization
            """
            batch_size, num_heads, seq_length, _ = output.shape
            
            head_means = torch.mean(output, dim=(2, 3), keepdim=True)
            
            attn_map_mean = mean_ratio * head_means + (1 - mean_ratio) * output
            
            return attn_map_mean
        
        def qkv_ifs(module, input, output, mean_ratio=self.r, smooth_component='q'):
            B, N, _ = output.shape
            # 拆分 Q/K/V 分量（核心步骤）
            D = output.shape[-1] // 3  # 单分量的维度
            q = output[:, :, 0:D]
            k = output[:, :, D:2*D]
            v = output[:, :, 2*D:3*D]
            
            # 仅对指定分量做均值平滑
            if smooth_component == 'q':
                q_mean = torch.mean(q, dim=1, keepdim=True)  # 计算Q的全局均值 [B,1,D]
                q = mean_ratio * q_mean + (1 - mean_ratio) * q  # 混合均值与原始Q
            elif smooth_component == 'k':
                k_mean = torch.mean(k, dim=1, keepdim=True)  # 计算K的全局均值
                k = mean_ratio * k_mean + (1 - mean_ratio) * k
            elif smooth_component == 'v':
                v_mean = torch.mean(v, dim=1, keepdim=True)  # 计算V的全局均值
                v = mean_ratio * v_mean + (1 - mean_ratio) * v
            else:
                raise ValueError("smooth_component 仅支持 'q'/'k'/'v'")
            
            # 重新拼接 Q/K/V 并返回
            qkv_processed = torch.cat([q, k, v], dim=-1)
            return qkv_processed
        
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


class IFS02(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.20, m=0.1, gamma=0.25, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)
    
    def _register_model(self):
        def attn_ifs(module, input, output, mean_ratio=self.r):
            """
            IGR: Intra-sample Global Regularization
            """
            batch_size, num_heads, seq_length, _ = output.shape
            
            head_means = torch.mean(output, dim=(2, 3), keepdim=True)
            
            attn_map_mean = mean_ratio * head_means + (1 - mean_ratio) * output
            
            return attn_map_mean
        
        def qkv_ifs(module, input, output, mean_ratio=self.r, smooth_component='k'):
            B, N, _ = output.shape
            # 拆分 Q/K/V 分量（核心步骤）
            D = output.shape[-1] // 3  # 单分量的维度
            q = output[:, :, 0:D]
            k = output[:, :, D:2*D]
            v = output[:, :, 2*D:3*D]
            
            # 仅对指定分量做均值平滑
            if smooth_component == 'q':
                q_mean = torch.mean(q, dim=1, keepdim=True)  # 计算Q的全局均值 [B,1,D]
                q = mean_ratio * q_mean + (1 - mean_ratio) * q  # 混合均值与原始Q
            elif smooth_component == 'k':
                k_mean = torch.mean(k, dim=1, keepdim=True)  # 计算K的全局均值
                k = mean_ratio * k_mean + (1 - mean_ratio) * k
            elif smooth_component == 'v':
                v_mean = torch.mean(v, dim=1, keepdim=True)  # 计算V的全局均值
                v = mean_ratio * v_mean + (1 - mean_ratio) * v
            else:
                raise ValueError("smooth_component 仅支持 'q'/'k'/'v'")
            
            # 重新拼接 Q/K/V 并返回
            qkv_processed = torch.cat([q, k, v], dim=-1)
            return qkv_processed
        
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


class IFS03(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.25, m=0.1, gamma=0.25, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)
    
    def _register_model(self):
        def attn_ifs(module, input, output, mean_ratio=self.r):
            """
            IGR: Intra-sample Global Regularization
            """
            batch_size, num_heads, seq_length, _ = output.shape
            
            head_means = torch.mean(output, dim=(2, 3), keepdim=True)
            
            attn_map_mean = mean_ratio * head_means + (1 - mean_ratio) * output
            
            return attn_map_mean
        
        def qkv_ifs(module, input, output, mean_ratio=self.r, smooth_component='v'):
            B, N, _ = output.shape
            # 拆分 Q/K/V 分量（核心步骤）
            D = output.shape[-1] // 3  # 单分量的维度
            q = output[:, :, 0:D]
            k = output[:, :, D:2*D]
            v = output[:, :, 2*D:3*D]
            
            # 仅对指定分量做均值平滑
            if smooth_component == 'q':
                q_mean = torch.mean(q, dim=1, keepdim=True)  # 计算Q的全局均值 [B,1,D]
                q = mean_ratio * q_mean + (1 - mean_ratio) * q  # 混合均值与原始Q
            elif smooth_component == 'k':
                k_mean = torch.mean(k, dim=1, keepdim=True)  # 计算K的全局均值
                k = mean_ratio * k_mean + (1 - mean_ratio) * k
            elif smooth_component == 'v':
                v_mean = torch.mean(v, dim=1, keepdim=True)  # 计算V的全局均值
                v = mean_ratio * v_mean + (1 - mean_ratio) * v
            else:
                raise ValueError("smooth_component 仅支持 'q'/'k'/'v'")
            
            # 重新拼接 Q/K/V 并返回
            qkv_processed = torch.cat([q, k, v], dim=-1)
            return qkv_processed
        
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


class IFS04(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.13, m=0.1, gamma=0.25, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)
    
    def _register_model(self):
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


class IFS05(IFS04):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.17, m=0.1, gamma=0.25, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class IFS06(IFS04):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.25, m=0.1, gamma=0.25, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class IFS07(IFS04):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.30, m=0.1, gamma=0.25, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)



class CEM(MIFGSM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, 
                 r=0.10, m=0.10, gamma=0.25,
                 targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, **kwargs)
        self.model_name     = model_name
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

        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            total_layers = 12
            for block_ind in range(total_layers):
                # cem
                layer_ratio = block_ind / (total_layers - 1)
                m_ratio = self.m + layer_ratio * self.gamma
                self.model.blocks[block_ind].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                self.model.blocks[block_ind].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
        elif self.model_name in ["pit_ti_224"]:
            total_layers = 12
            for block_ind in range(total_layers):
                layer_ratio = block_ind / (total_layers - 1)
                m_ratio = self.m + layer_ratio * self.gamma
                if block_ind < 2:
                    transformer_ind = 0
                    used_block_ind  = block_ind
                elif block_ind >= 2 and block_ind < 8:
                    transformer_ind = 1
                    used_block_ind  = block_ind - 2
                elif block_ind >= 8 and block_ind < 12:
                    transformer_ind = 2
                    used_block_ind  = block_ind - 8
                # cem
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
        elif self.model_name in ["cait_s24_224"]:
            total_layers = 26
            for block_ind in range(26):
                layer_ratio = block_ind / (total_layers - 1)
                m_ratio = self.m + layer_ratio * self.gamma 
                if block_ind < 24:
                    # cem
                    self.model.blocks[block_ind].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                    self.model.blocks[block_ind].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                else:
                    # cem
                    self.model.blocks_token_only[block_ind - 24].attn.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))
                    self.model.blocks_token_only[block_ind - 24].mlp.register_forward_hook(partial(cross_sample_embedding_mix, ratio=m_ratio))


class CEM01(CEM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.1, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class CEM02(CEM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.35, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class CEM03(CEM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.225, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)

# CEM
class SFM01(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.1, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class SFM02(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.15, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class SFM03(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.20, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class SFM04(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.25, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class SFM05(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.30, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class SFM06(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.35, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class SFM07(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.40, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)


class SFM08(SFM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, r=0.1, m=0.45, gamma=0.0, targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, r, m, gamma, targeted, random_start, norm, loss, device, **kwargs)

