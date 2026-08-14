import importlib

attack_zoo = {
    # gradient
    "mifgsm": (".gradient.mifgsm", "MIFGSM"),

    # model-related
    "pna_patchout": (".model_related.pna_patchout", "PNA_PatchOut"),
    "tgr": (".model_related.tgr", "TGR"),
    "gns": (".model_related.gns", "GNS"),
    "att": (".model_related.att", "ATT"),
    "fpr": (".model_related.fpr", "FPR"),

    # sfm
    "sfm": (".ours.sfm", "SFM"),

    # ifs
    "ifs": (".ours.sfm", "IFS"),
    "ifs01": (".ours.sfm", "IFS01"),
    "ifs02": (".ours.sfm", "IFS02"),
    "ifs03": (".ours.sfm", "IFS03"),
    "ifs04": (".ours.sfm", "IFS04"),
    "ifs05": (".ours.sfm", "IFS05"),
    "ifs06": (".ours.sfm", "IFS06"),
    "ifs07": (".ours.sfm", "IFS07"),

    # cem
    "cem": (".ours.sfm", "CEM"),
    "cem01": (".ours.sfm", "CEM01"),
    "cem02": (".ours.sfm", "CEM02"),
    "cem03": (".ours.sfm", "CEM03"),

    # sfm -- cem
    "sfm01": (".ours.sfm", "SFM01"),
    "sfm02": (".ours.sfm", "SFM02"),
    "sfm03": (".ours.sfm", "SFM03"),
    "sfm04": (".ours.sfm", "SFM04"),
    "sfm05": (".ours.sfm", "SFM05"),
    "sfm06": (".ours.sfm", "SFM06"),
    "sfm07": (".ours.sfm", "SFM07"),
    "sfm08": (".ours.sfm", "SFM08"),
}


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise Exception("Unsupported Attack Algorithm {}".format(attack_name))
    
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    attack_class = getattr(module, class_name)

    return attack_class