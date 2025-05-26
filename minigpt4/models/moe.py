import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
import copy
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners import lora

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


# # # loraMoE
class Expert(nn.Module, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):  
        super(Expert, self).__init__()
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.lora_num = lora_nums

        if r > 0:
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            for i in range(self.lora_num):
                setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
        self.scaling = self.lora_alpha / self.r

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)
        
        nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))
    
    def train(self, mode: bool = True):
        self.lora_route.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)
    
    def eval(self):
        self.lora_route.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            getattr(self, f"lora_B{i}").eval()
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)
    
    def forward(self, x: torch.Tensor):

        tmp = self.lora_route(x)
        topk_v, topk_idx =  torch.topk(tmp, 1, dim=-1)
        route_weight = torch.zeros_like(tmp)
        route_weight.scatter_(2, topk_idx, topk_v)


        results = torch.zeros(1)[0].to(x)
        for i in range(self.lora_num):
            results = results + torch.unsqueeze(route_weight[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling
        return results


class Moe(nn.Linear, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        task_nums: int = 10,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight
        self.task_nums = task_nums
        
        self.fan_in_fan_out = fan_in_fan_out
        self.criterion = nn.MSELoss()
        if r > 0:
            for i in range(self.task_nums):
                setattr(self, f"expert{i}", Expert(in_features, out_features, r, lora_alpha, lora_nums))
            setattr(self, f"shared_expert", Expert(in_features, out_features, r, lora_alpha, lora_nums))
            self.old_expert = None
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)


    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        for i in range(self.task_nums):
            getattr(self, f"expert{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        for i in range(self.task_nums):
            getattr(self, f"expert{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def before_train(self, task_id):
        if task_id == 0:
            return
        if task_id > 1:
            self.old_expert = copy.deepcopy(self.shared_expert)
        elif task_id == 1:
            self.old_expert = copy.deepcopy(self.expert0)
        for n, p in self.old_expert.named_parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, route_weight=None, train=False):
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        route_weight = route_weight.to(result.dtype)
        loss = torch.zeros(1)[0].to(result)

        if train:
            if self.task_nums > 0:
                b, n = route_weight.shape
                if n==1:
                    lamb = 1
                else:
                    lamb = 0.5

                if self.old_expert is not None:
                    shared_f = getattr(self, f"shared_expert")(x)
                    sheard_o = self.old_expert(x)
                    result = result + (1-lamb) * shared_f
                    loss += torch.nn.functional.mse_loss(shared_f, sheard_o)
                for i in range(n):
                    specific_f = getattr(self, f"expert{i}")(x)
                    result = result + lamb * route_weight[:,i].unsqueeze(-1).unsqueeze(-1) * specific_f
                    if n > 1 and i+1==n:
                        loss += torch.nn.functional.mse_loss(shared_f, specific_f.detach())
        else:
            if self.task_nums > 0:
                b, n = route_weight.shape
                if n==1:
                    lamb = 1
                else:
                    lamb = 0.5
                shared_f = getattr(self, f"shared_expert")(x)
                for i in range(n):
                    result = result + lamb * route_weight[:,i].unsqueeze(-1).unsqueeze(-1) * getattr(self, f"expert{i}")(x)
                result += (1-lamb) * shared_f
        return result, loss