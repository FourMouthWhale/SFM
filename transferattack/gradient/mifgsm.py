import torch

from ..utils import *
from ..attack import Attack


class MIFGSM(Attack):
    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., targeted=False, random_start=False, 
                 norm="linfty", loss="crossentropy", device=None, **kwargs):
        super().__init__(model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay