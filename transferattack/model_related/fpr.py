from functools import partial

import torch
import random
import numpy as np
import torch.nn.functional as F
from ..gradient.mifgsm import MIFGSM
from ..utils import *

accumulated_features = {}
class FPR(MIFGSM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, targeted=False,
                 random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, **kwargs)
        self.model_name = model_name
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
        def diverse_attn_map(module, input, output, attn_map_change_range):
            batch_size, num_heads, seq_length, _    = output.shape
            s_output                                = output * 1.0
            attn_map_noise                          = torch.empty_like(s_output)

            for head in range(num_heads):
                M               = torch.tensor(np.random.uniform(1 - attn_map_change_range, 1 + attn_map_change_range, (seq_length, seq_length)), dtype=torch.float32).to(output.device)
                noisy_attn      = s_output[:, head, :, :] * M
                normalized_attn = torch.softmax(noisy_attn, dim=-1)
                
                attn_map_noise[:, head, :, :]   = normalized_attn
            
            return attn_map_noise

        def cross_iter_emb_momentum(module, input, output, scale, mom_emb_decay):
            s_output    = output * scale
            module_id   = id(module)

            if module_id not in accumulated_features:
                accumulated_features[module_id] = s_output.clone()
            else:
                accumulated_features[module_id] = mom_emb_decay * accumulated_features[module_id].clone().detach() + s_output

            return accumulated_features[module_id]
        
        if self.model_name in ["vit_base_patch16_224"]:
            cr, s, d = 25, 0.8, 0.3
            for i in [0, 1, 4, 9, 11]:
                self.model.blocks[i].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=cr))

            for i in range(12):
                self.model.blocks[i].attn.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))    
                self.model.blocks[i].mlp.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
        elif self.model_name in ["deit_base_distilled_patch16_224"]: 
            cr, s, d = 30, 0.7, 0.2
            for i in [0, 1, 5, 10, 11]:
                self.model.blocks[i].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=cr)) 

            for i in range(12):
                self.model.blocks[i].attn.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
                self.model.blocks[i].mlp.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
        elif self.model_name in ["pit_ti_224"]: # 
            cr, s, d = 25, 0.7, 0.2            
            self.model.transformers[0].blocks[1].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=cr))
            self.model.transformers[1].blocks[4].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=cr))
            self.model.transformers[2].blocks[3].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=cr))
        
            for block_ind in range(12):                  
                if block_ind < 2:
                    transformer_ind = 0
                    used_block_ind = block_ind            
                elif block_ind < 8 and block_ind >= 2:
                    transformer_ind = 1
                    used_block_ind = block_ind - 2        
                elif block_ind < 12 and block_ind >= 8:
                    transformer_ind = 2
                    used_block_ind = block_ind - 8        
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
        elif self.model_name in ["cait_s24_224"]: 
            cr, s, d = 30, 0.6, 0.2
            for block_ind in [2, 14]:  
                self.model.blocks[block_ind].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=cr))
            for block_ind in [1]: 
                self.model.blocks_token_only[block_ind].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=cr))

            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
                    self.model.blocks[block_ind].mlp.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
                else:
                    self.model.blocks_token_only[block_ind - 24].attn.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
                    self.model.blocks_token_only[block_ind - 24].mlp.register_forward_hook(partial(cross_iter_emb_momentum, scale=s, mom_emb_decay=d))
