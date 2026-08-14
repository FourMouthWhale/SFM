import random
from functools import partial

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class PNA_PatchOut(MIFGSM):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1, ablation_study=['1', '1', '1'], 
                 targeted=False, random_start=False, norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, **kwargs)
        self.model_name     = model_name
        self.ablation_study = ablation_study

        # PNA (Pay No Attention)
        if self.ablation_study[0] == '1':
            self.model = self.model[1]
            self._register_model()
            self.model = wrap_model(self.model.eval().cuda())

        # PatchOut
        self.image_size         = 224
        self.crop_length        = 16
        self.max_num_patches    = int((224 / 16) ** 2)
        if self.ablation_study[1] == '1':
            self.sample_num_patches = 130
        else:
            self.sample_num_patches = self.max_num_patches
        assert self.sample_num_patches <= self.max_num_patches

        # L2 Norm
        if self.ablation_study[2] == '1':
            print("Using L2 Norm")
            self.lamb = 0.1
        else:
            print("Not Using L2 Norm")
            self.lamb = 0

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data    = data.clone().detach().to(self.device)
        label   = label.clone().detach().to(self.device)
        
        delta       = self.init_delta(data)
        momentum    = 0
        for epoch_idx in range(self.epoch):
            delta_patchout  = self._generate_samples_for_interactions(delta, seed=epoch_idx)
            logits          = self.get_logits(self.transform(data + delta_patchout))
            loss            = self.get_loss(logits, label)
            loss            += self.lamb * torch.norm(delta, p=2)
            grad            = self.get_grad(loss, delta)
            momentum        = self.get_momentum(grad, momentum)
            delta           = self.update_delta(delta, data, momentum, self.alpha)
        
        return delta.detach()

    def _register_model(self):
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:],)
        
        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

        if self.model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
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
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name in ["pit_ti_224"]:
            for block_ind in range(12):
                if block_ind < 2:
                    transformer_ind = 0
                    used_block_ind  = block_ind
                elif block_ind >= 2 and block_ind < 8:
                    transformer_ind = 1
                    used_block_ind  = block_ind - 2
                elif block_ind >= 8 and block_ind < 12:
                    transformer_ind = 2
                    used_block_ind  = block_ind - 8
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name in ["cait_s24_224"]:
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind > 24 and block_ind < 26:
                    self.model.blocks_token_only[block_ind - 24].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name in ["visfomer_small"]:
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >= 4 and block_ind < 8:
                    self.model.stage3[block_ind - 4].attn.attn_drop.register_backward_hook(drop_hook_func)
        
    def _generate_samples_for_interactions(self, delta, seed):
        add_noise_mask  = torch.zeros_like(delta)
        grid_num_axis   = int(self.image_size / self.crop_length)

        ids = [i for i in range(self.max_num_patches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_patches])

        rows, cols = ids // grid_num_axis, ids % grid_num_axis

        for r, c in zip(rows, cols):
            add_noise_mask[:, :, r * self.crop_length:(r + 1) * self.crop_length, c * self.crop_length:(c + 1) * self.crop_length] = 1
        add_perturbation = delta * add_noise_mask
        
        return add_perturbation
