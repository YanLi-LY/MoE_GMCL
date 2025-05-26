"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os



import random
# import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from continuum.tasks.image_path_task_set import PathTaskSet, ArrayTaskSet

from clip_base.datasets import build_cl_datasets, CustomPathTaskSet, CustomArrayTaskSet
import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    cfg_o = cfg.get_o_config()

    # wandb.init(project='gmm_moe', config=cfg_o, name=cfg_o.model.details)
    datasets, classes_names = build_cl_datasets(cfg_o, is_train=True)

    task_num = cfg_o.task_num
    model = task.build_model(cfg, train=True)
    total_classes = 0
    for i in range(task_num):
        
        inside_i = i

        # datasets = task.build_datasets(cfg, task_id=inside_i)
        # runner = get_runner_class(cfg)(
        #     cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets, task_id=inside_i
        # )
        if isinstance(datasets[i], PathTaskSet):
            custom_dataset = CustomPathTaskSet(datasets[i], classes_names)
        elif isinstance(datasets[i], ArrayTaskSet):
            custom_dataset = CustomArrayTaskSet(datasets[i], classes_names)
        
        runner = get_runner_class(cfg)(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets={'cc_sbu_align': {'train': custom_dataset}}, task_id=inside_i, total_classes = total_classes
        ) 
        runner.train()
        total_classes = runner.total_classes

    # wandb.finish()

if __name__ == "__main__":
    main()
