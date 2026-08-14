import torch
import torch.utils
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import timm
import os


img_height, img_width = 224, 224
img_max, img_min = 1., 0

model_to_name = {
    "resnet18"                          : "RN-18",
    "resnet50"                          : "Res-50",
    "resnet101"                         : "Res-101",
    "vgg16"                             : "VGG-16",
    "vgg19"                             : "VGG-19",
    "densenet121"                       : "DN-121",
    "efficientnet_b0"                   : "EN-b0",
    "tf_mobilenetv3_small_100"          : "MN-v3",
    "resnext50_32x4d"                   : "RNX-50",
    "inception_resnet_v2"               : "IncRes-v2",
    "resnetv2_152x2_bitm"               : "Res-v2",
    "ens_adv_inception_resnet_v2"       : "InRev2e",
    "inception_v3"                      : "Inc-v3",
    "adv_inception_v3"                  : "AdvIncv3",
    "inception_v4"                      : "Inc-v4",
    "cait_s24_224"                      : "CaiT-S",
    "coat_tiny"                         : "Coat-T", 
    "convit_base"                       : "ConViT-B",
    "deit_base_distilled_patch16_224"   : "Deit-B",
    "deit_tiny_patch16_224"             : "DeiT-T",
    "deit_small_patch16_224"            : "DeiT-S",
    "levit_256"                         : "LeViT-256",
    "pit_b_224"                         : "PiT-B",
    "pit_ti_224"                        : "PiT-T",
    "swin_tiny_patch4_window7_224"      : "Swin-T",
    "swin_small_patch4_window7_224"     : "Swin-S",
    'tnt_s_patch16_224'                 : "TNT-S",
    "visformer_small"                   : "Vis-S",
    "visformer_tiny"                    : "Vis-T",
    "vit_base_patch16_224"              : "ViT-B",
}


source_cnn_models = []
# source_vit_models = ["vit_base_patch16_224", "deit_base_distilled_patch16_224", "pit_ti_224", "cait_s24_224"]
source_vit_models = ["vit_base_patch16_224"]

target_torch_models = []
target_timm_models = ["vit_base_patch16_224", "deit_base_distilled_patch16_224", "pit_ti_224", 
                      "cait_s24_224", "pit_b_224", "visformer_small", "swin_tiny_patch4_window7_224", 
                      "coat_tiny", "levit_256", "tnt_s_patch16_224", "convit_base", 
                      "inception_v3", "inception_v4", "inception_resnet_v2", "resnet50", "resnetv2_152x2_bitm"]


generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]


def load_pretrained_model(target_torch_models=[], target_timm_models=[]):
    for model_name in target_torch_models:
        yield model_name, models.__dict__[model_name](weights="DEFAULT")
    for model_name in target_timm_models:
        if model_name == "levit_256":
            yield model_name, timm.create_model(model_name, pretrained=True, 
                                                pretrained_cfg_overlay=dict(file='./checkpoints/levit_256.bin'))
        else:
            yield model_name, timm.create_model(model_name, pretrained=True)

"""
        If you encounter model loading issues about levit_256, 
        you can download the checkpoint 'pytorch_model.bin' from https://huggingface.co/timm/levit_256.fb_dist_in1k/tree/main.

        if model_name == "levit_256":
            yield model_name, timm.create_model(model_name, pretrained=True, 
                                                pretrained_cfg_overlay=dict(file='./transferattack/checkpoints/levit_256.bin'))
        else:
            yield model_name, timm.create_model(model_name, pretrained=True)
"""

def wrap_model(model):
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.data_dir = input_dir
        self.targeted = targeted
        self.target_class = target_class
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        if eval:
            self.data_dir = output_dir
            print("=> Eval mode: evaluating on {}".format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Generating mode: Generating on {}'.format(self.data_dir))
            print("Save images to {}".format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        image = np.array(image).astype(np.float32) / 255
        image = torch.from_numpy(image).permute(2, 0, 1)

        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        """
        targeted == True: return {"filename": ["label", "targeted_label"]}
        targeted == False: return {"filename": "label"}
        """
        dev = pd.read_csv(file_name)
        if self.targeted:
            if self.target_class:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], self.target_class] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
        
        return f2l