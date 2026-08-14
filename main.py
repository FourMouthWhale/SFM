import os
import yaml
import torch
import tqdm
import random
import transferattack
from transferattack.utils import *

import warnings
warnings.filterwarnings("ignore")

from time import time


def main(config):
    # ======================= Config =======================
    source_model = config['model']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU_ID']
    batch_size = config['batch_size']

    # ===================== DataLoader =====================
    dataset     = AdvDataset(input_dir=config['input_dir'], output_dir=config['output_dir'], targeted=config['targeted'], eval=config['eval'])
    dataloader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'])

    # ============ Generate Adversarial Examples ===========
    if not config['eval']:
        if config['ensemble'] or len(source_model.split(',')) > 1:
            source_model = source_model.split(',')
        attacker = transferattack.load_attack_class(config['attack'])(model_name=source_model, targeted=config['targeted'])

        for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
            perturbations = attacker(images, labels)
            save_images(config['output_dir'], images + perturbations.cpu(), filenames)
    # ======================== Eval ========================
    else:
        res = '|'
        total_asr = 0
        for model_name, model in load_pretrained_model(target_torch_models, target_timm_models):
            model = wrap_model(model.eval().cuda())
            for p in model.parameters():
                p.requires_grad = False

            asr = eval_attack(model, dataloader, config['targeted'])
            print(f"{model_to_name[model_name].center(10)}: {asr:.1f}")
            res += f"   {asr:.1f}   |"
            total_asr += asr
        avg_asr = total_asr / (len(target_torch_models + target_timm_models))
        print(f"Average ASR: {avg_asr}")
        res += f"   {avg_asr:.1f}   |"
        print(res)

        with open('results_eval.txt', 'a') as f:
            centered_attack = config['attack'] + "->" + model_to_name[source_model]
            centered_attack = centered_attack
            f.write(centered_attack.center(24) + res + '\n')


def eval_attack(model, dataloader, is_targeted: bool):
    correct, total = 0, 0
    for images, labels, _ in dataloader:
        if is_targeted:
            labels = labels[1]
        pred = model(images.cuda())
        correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
        total += labels.shape[0]
    if is_targeted:
        # correct: pred == target_label
        asr = (correct / total) * 100
    else:
        # correct: pred == original_label
        asr = (1 - correct / total) * 100
    return asr

def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main_attack_and_eval():
    set_randomseed(seed=42)
    # "mifgsm", "pna_patchout", "tgr", "gns", "att", "fpr", "sfm"
    attack_list = ["sfm"]

    for attack_idx, attack in enumerate(attack_list):
        for model_idx, model_name in enumerate(source_vit_models):
            # config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            config['attack']        = attack
            config['model']         = model_name
            config['output_dir']    = os.path.join(config['output_dir'], attack, model_to_name[config['model']])
            if attack in ["mifgsm", "sfm"]:
                config['batch_size'] = 16
            elif attack in ["fpr"]:
                config['batch_size'] = 8
            elif attack in ['pna_patchout', "tgr", "gns", "att"]:
                config['batch_size'] = 1
            else:
                config['batch_size'] = 16
            
            if not os.path.exists(config['output_dir']):
                os.makedirs(config['output_dir'])
                
            print(f"============Attack: {attack.center(10)}============")
            if config['eval']:
                eval_models = " " * 24 + "| "
                for i in target_torch_models + target_timm_models:
                    eval_models += model_to_name[i].center(8) + " | "
                if attack_idx == 0 and model_idx == 0:
                    with open('results_eval.txt', 'a') as f:
                        f.write(eval_models + '\n')
            
            # Generate Adversarial Examples using Attack Method
            start   = time()
            main(config)
            end     = time()
            print(f"Attack Time: {end - start}")

            # Eval
            config['eval']          = True
            config['batch_size']    = 32
            main(config)


if __name__ == "__main__":
    main_attack_and_eval()