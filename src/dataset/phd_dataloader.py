#!/usr/bin/env python3

from typing import Any, Callable, Dict, Optional, Union

from classy_vision.dataset import ClassyDataset, register_dataset
from classy_vision.dataset.transforms import ClassyTransform, build_transforms
from classy_vision.models.T5.utils import *
from classy_vision.models.T5.utils.constants import *
from .dataset_co_model_lvtr_offline import DatasetCoModelLvtrOffline
import json
import torch
from classy_vision.models.bert import BertTokenizer
import random
import numpy as np
import PIL.Image as Image
import logging
import time

from types import SimpleNamespace as ConfigNode

random.seed(100)

@register_dataset("phd_dataloader")
class phd_dataloader(ClassyDataset):

    def __init__(
        self,
        meta_file: str,
        history_root_path: str,
        history_root_path_yolo: str,
        dirs: str,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Optional[Union[ClassyTransform, Callable]],
        num_samples: Optional[int],
        tokenizer: Optional[BertTokenizer],
        token_max_length: Optional[int],
        drop_ratio: float,
        target_len: int,
        cls_num: int,
        mapping: str,
        num_pids: int,
        live_mean_ctr_file:str,
        smooth_ctr: bool,
        img_path: str,
        use_pxtr: bool,
        mxnet_path: str,
        use_author_embed: bool
        
    ):
        self.dataset = DatasetCoModelLvtrOffline(meta_file,history_root_path,history_root_path_yolo, dirs, num_samples, num_pids=num_pids,live_mean_ctr_file=live_mean_ctr_file,\
            smooth_ctr=smooth_ctr,img_path=img_path,use_pxtr=use_pxtr,\
                mxnet_path=mxnet_path)
        super().__init__(
            self.dataset, batchsize_per_replica, shuffle, transform, num_samples
        )
        self.tokenizer = tokenizer
        self.token_max_length = token_max_length
        self.drop_ratio = drop_ratio
        self.target_max_length = target_len
        self.cls_num = cls_num
        self.asr_text_max=100
        self.comment_text_max=self.token_max_length-self.asr_text_max-3
        self.text_tokens=200
        self.caption_text_max = 200
        self.image_tokens = 1
        self.use_author_embed=use_author_embed
        self.text_align_max = 200
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Class_MutilModal_Seq2Seq":

        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
        ) = cls.parse_config(config)
        meta_file = config.get('meta_file')
        history_root_path = config.get('history_root_path')
        history_root_path_yolo = config.get('history_root_path_yolo')
        dirs = config.get('dirs')
        drop_ratio = config.get('drop_ratio')
        target_len = config.get("target_len")
        cls_num = config.get('cls_num')
        mapping = config.get('mapping')
        num_pids=config.get('number_pids')
        smooth_ctr=config.get('smooth_ctr',False)
        img_path=config.get('img_path','./imgs')
        live_mean_ctr_file = config.get('live_mean_ctr_file','')
        use_pxtr=config.get('use_pxtr',False)
        mxnet_path = config.get('mxnet_path',"")
        use_author_embed=config.get('use_author_embed',False)

        tokenizer_config = config.get('tokenizer', '')
        if tokenizer_config != '':
            tokenizer_config = ConfigNode(**tokenizer_config)
            tokenizer = str2tokenizer[tokenizer_config.type](tokenizer_config)
        else:
            tokenizer = ''
            token_max_length = 0
        transform = build_transforms(transform_config)
        return cls(
            meta_file = meta_file,
            history_root_path  = history_root_path,
            history_root_path_yolo = history_root_path_yolo,
            dirs = dirs,
            batchsize_per_replica = batchsize_per_replica,
            shuffle = shuffle,
            transform = transform,
            num_samples = num_samples,
            tokenizer = tokenizer,
            token_max_length = tokenizer_config.token_max_length,
            drop_ratio = drop_ratio,
            target_len = target_len,
            cls_num = cls_num,
            mapping = mapping,
            num_pids=num_pids,
            live_mean_ctr_file=live_mean_ctr_file,
            smooth_ctr=smooth_ctr,
            img_path=img_path,
            use_pxtr=use_pxtr,
            mxnet_path=mxnet_path,
            use_author_embed=use_author_embed
        )
        
    def __getitem__(self, idx: int):
        assert idx >= 0 and idx < len(
            self.dataset
        ), "Provided idx ({}) is outside of dataset range".format(idx)
        sample = self.dataset[idx]
    
        sample_dict = self.dataset[idx]
        final_images=sample_dict['final_images'] 
        final_caption_text = sample_dict['final_caption_texts']  
        target=sample_dict['unid_target'] 
        gt_nframes_binary = sample_dict['gt_nframes_binary']
        youtube_id = sample_dict['youtube_id']
        user_history_collection = sample_dict['user_history_collection']
        user_history_collection_yolo = sample_dict['user_history_collection_yolo']

        gt_nframes = sample_dict['gt_nframes']

        sample=[final_images]

        images_count = len(sample[0])
        pad_id = self.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
        texts, segs, decoder_src_inputs = [], [], []
        tokenized_text_align = []
        if self.tokenizer != '':
            for cur_caption in final_caption_text:
                cur_caption_token = self.tokenizer.tokenize(cur_caption)
                if len(cur_caption_token)>self.caption_text_max :
                    cur_caption_token = cur_caption_token[:self.caption_text_max]
                
                tokenized_text = ["[CLS]"] + ["[SEP]"]+cur_caption_token  # cls img sep asr seq comm pad
                indexed_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                indexed_text = indexed_text[:self.token_max_length]
                input_mask = [1] * len(indexed_text)

                padding = [pad_id] * (self.token_max_length - len(indexed_text))
                indexed_text += padding
                indexed_text = indexed_text[:self.token_max_length]
                texts.append(indexed_text)
                input_mask += padding
                input_mask = input_mask[:self.token_max_length]
                if self.use_author_embed:
                    seg = [1] * (self.image_tokens+1) + input_mask
                else:
                    seg = [1] * self.image_tokens + input_mask
                segs.append(seg)


                if indexed_text not in tokenized_text_align:
                    tokenized_text_align.append(indexed_text)
                    segs.append(seg)

        for indexed_text in tokenized_text_align:
            texts.append(indexed_text)


        if self.transform is not None:
            sample = self.transform(sample)

        pos_id = len(final_caption_text)*[list(range(self.image_tokens+self.text_tokens))]

        user_history_collection = torch.tensor(user_history_collection)
        user_history_collection_yolo = torch.tensor(user_history_collection_yolo)

        sample['target'] = torch.tensor(target)
        sample['text_feature'] = torch.tensor(texts)
        sample['seg'] = torch.tensor(segs)
        sample['user_history_collection'] = user_history_collection
        sample['user_history_collection_yolo'] = user_history_collection_yolo
        sample['youtube_id'] = youtube_id
        sample['pos_id']= torch.tensor(pos_id)
        sample['task'] = torch.tensor(1)
        sample['gt_nframes_binary'] = torch.tensor(gt_nframes_binary)
        sample['gt_nframes'] = gt_nframes

        return sample
