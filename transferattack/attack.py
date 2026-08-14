import torch
import torch.nn as nn

import numpy as np

from .utils import *

class Attack(object):
    """
    Base class for all attacks
    """
    def __init__(self, model_name: str, epsilon: float, targeted: bool, 
                 random_start: bool, norm: str, loss: str, device=None):
        """
        Initialize the hyperparameters

        Arguments:
            attack: the name of attack
            model_name: the name of surrogate model
            epsilon: the pertrubation budget
            targeted: targeted/untargeted attack
            random_start: whether using random initialization for delta
            norm: the norm of perturbation, l2 / linfty
            loss: the loss function
            device: the device for data. If it is None, the device would be same as model
        """
        if norm not in ["l2", "linfty"]:
            raise Exception("Unsupported norm {}".format(norm))
        self.model = self.load_model(model_name)
        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        if isinstance(self.model, EnsembleModel):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device if device is None else device
        self.loss = self.loss_function(loss)


    def load_model(self, model_name) -> nn.Module:
        """
        Model Loading Stage

        Arguments:
            model_name: the name of surrogate model

        Returns:
            model: the surrogate model wrapped by wrap_model in utils.py
        """
        def load_single_model(model_name):
            if model_name in timm.list_models():
                print("=> Loading model {} from timm.models".format(model_name))
                model = timm.create_model(model_name, pretrained=True)
            elif model_name in models.__dict__.keys():
                print("=> Loading model {} from torchvision.models".format(model_name))
                model = models.__dict__[model_name](weights="DEFAULT")
            else:
                raise ValueError("Model {} is not supported".format(model_name))
            return wrap_model(model.eval().cuda())
        
        if isinstance(model_name, list):
            return EnsembleModel([load_single_model(name) for name in model_name])
        else:
            return load_single_model(model_name)

    def forward(self, data: torch.tensor, label: torch.tensor, **kwargs) -> torch.tensor:
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W), tensor for input images
            label: (N,), tentor for ground-truth labels if untargeted
            label: (2, N), tentor for [ground-truth, targeted label] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data + delta, momentum=momentum))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        
        return delta.detach()

    def get_logits(self, x, **kwargs):
        """
        The inference stage
        """
        return self.model(x)

    def get_loss(self, logits, label):
        """
        The loss calculation
        """
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)

    def get_grad(self, loss: torch.tensor, delta: torch.tensor, **kwargs) -> torch.tensor:
        """
        The gradient calculation
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad, momentum):
        """
        The momentum calculation
        """
        return self.decay * momentum + grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True)

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == "linfty":
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0, 1).to(self.device)
                delta *= r / n * self.epsilon
            delta = clamp(delta, img_min - data, img_max - data) 
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == "linfty":
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min - data, img_max - data)
        return delta.detach().requires_grad_(True)

    def loss_function(self, loss):
        if loss == "crossentropy":
            return nn.CrossEntropyLoss()
        else:
            raise Exception("Unsupported Loss {}".format(loss))

    def transform(self, data, **kwargs):
        """
        Input transformation
        """
        return data

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)