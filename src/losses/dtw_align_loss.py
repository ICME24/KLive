#!/usr/bin/env python3
from typing import Any, Dict

import torch
import torch.nn as nn

from . import ClassyLoss, build_loss, register_loss
from classy_vision.generic.distributed_util import gather_from_all
import json
import traceback
import os
import torch.nn.functional as F
import time
import torch
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import random
import itertools
import time



def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')

def DTW_cum_dist(dists, lbda=0.1):
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    cum_dists[:, :, 0, 0] = dists[:, :, 0, 0]

    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 0] = dists[:, :, l, 0] + cum_dists[:, :, l - 1, 0]

    for l in range(1, dists.shape[2]):
        for m in range(1, dists.shape[3]):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.logsumexp(
                torch.cat(
                    [
                        -cum_dists[:, :, l - 1, m - 1].unsqueeze(dim=-1) / lbda,
                        -cum_dists[:, :, l, m - 1].unsqueeze(dim=-1) / lbda,
                        -cum_dists[:, :, l - 1, m].unsqueeze(dim=-1) / lbda,
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

    return cum_dists[:, :, -1, -1]

def OTAM_cum_dist(dists, lbda=0.1):
    dists = F.pad(dists, (1, 1), "constant", 0)

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] - lbda * torch.logsumexp(
            -cum_dists[:, :, 0, m - 1].unsqueeze(dim=-1) / lbda, dim=-1
        )

    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.logsumexp(
            torch.cat(
                [
                    -cum_dists[:, :, l - 1, 0].unsqueeze(dim=-1) / lbda,
                    -cum_dists[:, :, l - 1, 1].unsqueeze(dim=-1) / lbda,
                    -cum_dists[:, :, l, 0].unsqueeze(dim=-1) / lbda,
                ],
                dim=-1,
            ),
            dim=-1,
        )

        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.logsumexp(
                torch.cat(
                    [
                        -cum_dists[:, :, l - 1, m - 1].unsqueeze(dim=-1) / lbda,
                        -cum_dists[:, :, l, m - 1].unsqueeze(dim=-1) / lbda,
                    ],
                    dim=-1,
                ),
                dim=-1,
            )

        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.logsumexp(
            torch.cat(
                [
                    -cum_dists[:, :, l - 1, -2].unsqueeze(dim=-1) / lbda,
                    -cum_dists[:, :, l - 1, -1].unsqueeze(dim=-1) / lbda,
                    -cum_dists[:, :, l, -2].unsqueeze(dim=-1) / lbda,
                ],
                dim=-1,
            ),
            dim=-1,
        )

    return cum_dists[:, :, -1, -1]



def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

def sample_from_dist(sim_tensor, gamma = 0.1):
    for i in range(sim_tensor.size(0)):
        sim_tensor[i, i] = float('-inf')
        for j in range(i):
            sim_tensor[i, j] = float('-inf')
    values, indices = torch.sort(sim_tensor.view(-1), descending=True)
    temp_weight = F.softmax(values / gamma, dim=0)

    row_indices = indices // sim_tensor.size(1)
    col_indices = indices % sim_tensor.size(1)

    sample = random.choices(values, temp_weight)[0]
    index = torch.where(values == sample)[0][0]


    return row_indices[index].cpu().long().tolist(), col_indices[index].cpu().long().tolist()




def generate_alignment_keys(
        pooled_video, pooled_text, seq_length, total_sample_size
    ):
        seq_total_length = seq_length
        threshold = seq_length
        indices = torch.tensor([i for i in range(seq_total_length)])
        idx_f = np.linspace(0, len(indices) - 1, num=threshold)
        idxs = [int(f) for f in idx_f]
        sequence = indices[torch.tensor(idxs)].to(pooled_video.device)

        pooled_video = torch.index_select(pooled_video, 0, sequence)
        pooled_text = torch.index_select(pooled_text, 0, sequence)

        sim = cos_sim(pooled_text, pooled_video)
        detached_sim = sim.clone().detach()
        index_i, index_j = sample_from_dist(detached_sim)
        orig_dists = 1 - sim
        dists = rearrange(orig_dists, "(b s) d -> b s d", b=1)

        negative_keys = []
        indices = [i for i in range(orig_dists.shape[-1])]
        indices[index_i] = index_j
        indices[index_j] = index_i
        perms = []
        cnt = 0
        for i in range(total_sample_size):
            perm = np.random.permutation(indices)
            if list(perm) not in perms and list(perm) != indices:
                cnt += 1
                perms.append(list(perm))
        random.shuffle(perms)
        indices = torch.tensor(indices).to(pooled_video.device)
        for perm in perms:
            negative_sequence = torch.tensor(perm).to(pooled_video.device)
            if not torch.equal(indices, negative_sequence):
                negative_key = torch.index_select(orig_dists, 0, negative_sequence)
                negative_key = rearrange(negative_key, "(b s) d -> b s d", b=1)
                negative_keys.append(negative_key)
            if len(negative_keys) >= total_sample_size - 1:
                break

        keys = torch.stack([dists] + negative_keys, dim=1)
        return keys


def generate_alignment_loss(logits, total_sample_size):
        criterion = torch.nn.CrossEntropyLoss()
        alignment_losses = [
            criterion(
                logits[:, i * total_sample_size : (i + 1) * total_sample_size],
                torch.tensor([0]).to(logits.device),
            )
            for i in range(logits.shape[1] // total_sample_size)
        ]
        alignment_loss = sum(alignment_losses)
        return alignment_loss




@register_loss("dtw_align_loss")
class dtw_align_loss(ClassyLoss):
    def __init__(self, num_classes, target_len, num_tasks=1, log_name=None, batch=2,use_mean_ctr=False,diff_weight=1,diff_loss='mse',\
        unit_ctr_weight=1,live_mean_weight=1,live_mean_loss_name='bce',use_listwise=False,listwise_softmax_temp=5,list_loss_weight=1,use_pxtr=False,\
            predict_next=False,all_unit_bce_loss=False):
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss(reduction='none')
        self.count = 0
        self.ce = nn.CrossEntropyLoss()
        
        self.margin = 0.2
        self.batch = batch
        self.unit_ctr_bce_loss =nn.BCEWithLogitsLoss()
        self.unit_ctr_weight = unit_ctr_weight
        self.total_sample_size_align = 8
        
        self.use_mean_ctr = use_mean_ctr
        
       
        if live_mean_loss_name=="mse":
            self.live_mean_ctr_loss = nn.MSELoss()
        elif live_mean_loss_name=="l1":
            self.live_mean_ctr_loss = nn.L1Loss()
        elif live_mean_loss_name=='bce':
            self.live_mean_ctr_loss =nn.BCEWithLogitsLoss()
        
        self.live_mean_loss_name = live_mean_loss_name
        self.live_mean_weight = live_mean_weight

        self.diff_weight = diff_weight
        if diff_loss=="mse":
            self.diff_loss = nn.MSELoss()
        elif diff_loss=="l1":
            self.diff_loss = nn.L1Loss()
        
        self.use_listwise = use_listwise
        self.listwise_softmax_temp = listwise_softmax_temp
        self.list_cross_loss=nn.CrossEntropyLoss()
        self.list_loss_weight = list_loss_weight
        self.use_pxtr=use_pxtr
        self.predict_next = predict_next
        self.all_unit_bce_loss=all_unit_bce_loss

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiLevelSumLoss":
        """Instantiates a MultiOutputSumLoss from a configuration.

        Args:
            config: A configuration for a MultiOutpuSumLoss.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A MultiOutputSumLoss instance.
        """
        return cls(num_classes=config.get("class_num"), target_len=config.get("target_len"), 
                   num_tasks=config.get("num_tasks"), log_name=config.get("log_name", None),
                   batch=config.get("batch", 2),\
                    use_mean_ctr=config.get("use_mean_ctr", False),
                    diff_weight=config.get("diff_weight", 1),
                    diff_loss=config.get("diff_loss", "mse"),
                    unit_ctr_weight=config.get("unit_ctr_weight", 1),
                    live_mean_weight=config.get("live_mean_weight", 1),
                    live_mean_loss_name=config.get("live_mean_loss_name", 'bce'),
                    listwise_softmax_temp = config.get("listwise_softmax_temp", 5),
                    use_listwise = config.get("use_mean_ctr", False),
                    list_loss_weight=config.get("list_loss_weight", 0),
                    use_pxtr=config.get("use_pxtr", False),
                    predict_next=config.get("predict_next", False),
                    all_unit_bce_loss=config.get("all_unit_bce_loss", False)
                    )

    def update_weights(self, target, output, task_id):
        batch = target.shape[0]
        prediction = output.softmax(-1).argmax(-1)
        for i in range(self.target_len):
            keeps = prediction[:, i] != 0
            correct = (target[keeps, i] == prediction[keeps, i]).sum()
            self.correct_samples[task_id][i] += correct.item()
            self.total_samples[task_id][i] += batch
            self.weights[task_id][i] = 1 - self.correct_samples[task_id][i] / self.total_samples[task_id][i]
        self.first_weights[task_id] = self.weights[task_id][0].item()


    def forward(self, output, target):
        highlight_logit = output['highlight_logit']
        text_align_emb = output['text_align_emb']
        image_align_emb = output['image_align_emb']

        input_tensor = highlight_logit.contiguous().view(-1,2)
        target = target.view(-1)

        loss_classification = F.cross_entropy(input_tensor, target, weight=None, ignore_index=-100, reduction='mean')


        alignment_keys = generate_alignment_keys(image_align_emb, text_align_emb, len(image_align_emb),  total_sample_size=self.total_sample_size_align)

        cum_dists = DTW_cum_dist(alignment_keys)

        logits = -cum_dists

        alignment_loss = 0.01 * generate_alignment_loss(logits, total_sample_size=self.total_sample_size_align)

        loss = loss_classification + alignment_loss

        if not torch.is_tensor(alignment_loss):
            alignment_loss = torch.tensor(alignment_loss)

        final_res = [loss, loss_classification, alignment_loss]

        return final_res