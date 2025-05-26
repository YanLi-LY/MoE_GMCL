import os
import json
import yaml
from omegaconf import DictConfig, OmegaConf


def get_class_order(file_name: str) -> list:
    r"""TO BE DOCUMENTED"""
    with open(file_name, "r+") as f:
        data = yaml.safe_load(f)
        return data["class_order"]

def get_ordered_class_name(class_order, class_name):
    new_class_name = []
    for i in range(len(class_name)):
        new_class_name.append(class_name[class_order[i]])
    return new_class_name

def get_class_ids_per_task(args):
    yield args.class_order[:args.initial_increment]
    for i in range(args.initial_increment, len(args.class_order), args.increment):
        yield args.class_order[i:i + args.increment]

def get_imagnet1k_class_names(path):
    with open(os.path.join(path, "classnames.txt"), "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]


def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")


def get_workdir(path):
    split_path = path.split("/")
    workdir_idx = split_path.index("clip_based")
    return "/".join(split_path[:workdir_idx+1])

def get_tinyimagenet_classnames(path):
    classnames = []
    words_file = os.path.join(path, "words.txt")
    wnids_file = os.path.join(path, "wnids.txt")
    
    set_nids = []
    
    with open(wnids_file, 'r') as fo:
        data = fo.readlines()
        for entry in data:
            set_nids.append(entry.strip("\n"))
    
    class_maps = {}
    with open(words_file, 'r') as fo:
        data = fo.readlines()
        for entry in data:
            words = entry.split("\t")
            if words[0] in set_nids:
                class_maps[words[0]] = (words[1].strip("\n").split(","))[0]
    for data in set_nids:
        classnames.append(class_maps[data])
    return classnames

def get_cub200_classnames(path):
    classnames = []
    classes_txt = os.path.join(path, 'classes.txt')

    with open(classes_txt, 'r') as fo:
        data = fo.readlines()
        for entry in data:
            words = entry.split(".")
            classnames.append(words[1].strip("\n").replace('_', ' '))

    return classnames