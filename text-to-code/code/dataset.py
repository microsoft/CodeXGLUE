# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import gc
import shutil
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


class concodeDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=512, mode='train'):
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()

        self.block_size = block_size
        self.mode = mode

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank))
        if mode != 'test' and os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                data = pickle.load(handle)
                self.inputs = data['inputs']
                self.token_labels = data['token_labels']

        else:
            self.inputs = []
            self.token_labels = []

            datafile = os.path.join(args.data_dir, f"{file_type}.json")
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)
            datas = open(datafile).readlines()

            length = len(datas)
            logger.info("Data size: %d"%(length))
            for idx, x in enumerate(datas):
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))
                if idx % world_size != local_rank:
                    continue
                x = json.loads(x)
                code = tokenizer.encode(x["code"])
                nl = tokenizer.encode(x["nl"])

                input_ids, input_labels = self.pad_and_get_mask(code, nl, tokenizer)
                self.inputs.append(input_ids)
                self.token_labels.append(input_labels)

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples"%(local_rank, length, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            if mode != 'test':
                with open(cached_file, 'wb') as handle:
                    pickle.dump({'inputs': self.inputs, 'token_labels': self.token_labels}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def pad_and_get_mask(self, code, nl, tokenizer):
        if self.mode == 'test':
            code = []
        while (len(code) + len(nl) + 2 > self.block_size):
            if (len(code) > len(nl)):
                code = code[:-1]
            else:
                nl = nl[:-1]
        if self.mode == 'train':
            inputs = nl + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
            labels = [1] * len(nl) + [2] * (len(code)+1) + [0]
        else:
            inputs = nl + [tokenizer.bos_token_id]
            labels = [1] * len(nl) + [2]
            return inputs, labels
        assert len(inputs) <= self.block_size
        pad_len = self.block_size - len(inputs)
        inputs += [tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)
        return inputs, labels


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])
