# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
from bleu import _bleu
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'c_sharp':DFG_csharp,
}

def get_example(item):
    source,target,tokenizer,args,parser=item
    code=source.replace('. ','.')
    nl=target.strip()
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    
    #obtain dataflow
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        codes=code_tokens
        dfg=new_DFG
    except:
        codes=code.split()
        dfg=[]
    #merge nodes
    dic={}
    for d in dfg:
        if d[1] not in dic:
            dic[d[1]]=d
        else:
            dic[d[1]]=(d[0],d[1],d[2],list(set(dic[d[1]][3]+d[3])),list(set(dic[d[1]][4]+d[4])))
    DFG=[]
    for d in dic:
        DFG.append(dic[d])
    dfg=DFG
    return convert_examples_to_features(codes,dfg,nl,tokenizer,args)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 target_tokens,
                 target_ids,                 
                 dfg_tokens,
                 dfg_ids,
                 dfg_to_code,
                 dfg_to_dfg,
                 target,


    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.target_tokens=target_tokens
        self.target_ids=target_ids
        self.dfg_tokens=dfg_tokens
        self.dfg_ids=dfg_ids
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg
        self.target=target
        
def convert_examples_to_features(code,dfg,nl,tokenizer,args):
    code_tokens=[tokenizer.tokenize(x) for x in code]
    dfg=dfg[:args.max_dfg_length]
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_tokens=[x[0] for x in dfg]
    dfg_to_dfg=[x[-1] for x in dfg]
    dic={}
    dic[-1]=(0,0)
    for i in range(len(code_tokens)):
        dic[i]=(dic[i-1][1],dic[i-1][1]+len(code_tokens[i]))
    dfg_to_code=[dic[x[1]] for x in dfg] 
    
    
    code_tokens=[y for x in code_tokens for y in x][:args.max_source_length-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.max_source_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    
    nl_tokens=tokenizer.tokenize(nl)[:args.max_target_length-2]
    target_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    target_ids =  tokenizer.convert_tokens_to_ids(target_tokens)
    padding_length = args.max_target_length - len(target_ids)
    target_ids+=[tokenizer.pad_token_id]*padding_length    
    
    
    index=1
    dfg_ids=tokenizer.convert_tokens_to_ids(dfg_tokens)
    dfg_to_code=[(x[0]+index,x[1]+index) for x in dfg_to_code]
    padding_length = args.max_dfg_length - len(dfg_ids)
    dfg_ids+=[tokenizer.pad_token_id]*padding_length
    
    return InputFeatures(source_tokens,source_ids,target_tokens,target_ids,
                         dfg_tokens,dfg_ids,dfg_to_code,dfg_to_dfg,nl)




class TextDataset(Dataset):
    def __init__(self, tokenizer, args, source_file_path,target_file_path):
        self.args=args
        #load dataflow parser
        LANGUAGE = Language('parser/my-languages.so', args.language)
        parser = Parser()
        parser.set_language(LANGUAGE) 
        parser = [parser,dfg_function[args.language]]
        
        #get examples
        self.examples = []
        with open(source_file_path) as f1,open(target_file_path) as f2:
            for line1,line2 in zip(f1,f2):
                self.examples.append((line1,line2,tokenizer, args,parser))
                
        #extract data flow and tokenize            
        self.examples=[get_example(x) for x in tqdm(self.examples,total=len(self.examples))]
        

        if 'train' in source_file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))    
                    logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in example.target_tokens]))
                    logger.info("target_ids: {}".format(' '.join(map(str, example.target_ids))))                     
                    logger.info("dfg_tokens: {}".format(example.dfg_tokens))
                    logger.info("dfg_ids: {}".format(' '.join(map(str, example.dfg_ids))))
                    logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                    logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))
        rate=[]
        for example in self.examples:
            rate.append(len(example.dfg_tokens)!=0)
        logger.info("DFG Rate: %s",round(np.mean(rate),4))
        """
        dfg_to_code_mask: shape(#nodes,#input), 1 means edge between nodes and codes
        dfg_to_dfg_mask: shape(#nodes,#node), 1 means edge between nodes and nodes
        """
        self.dfg_to_code_mask=np.zeros((len(self.examples),args.max_dfg_length,args.max_source_length),dtype=np.bool)
        self.dfg_to_dfg_mask=np.zeros((len(self.examples),args.max_dfg_length,args.max_dfg_length),dtype=np.bool)
        self.tag=[False for x in range(len(self.examples))]
        
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.tag[item]:
            dfg_to_code_mask=self.dfg_to_code_mask[item]
            dfg_to_dfg_mask=self.dfg_to_dfg_mask[item]
        else:
            dfg_to_code_mask=self.dfg_to_code_mask[item]
            dfg_to_dfg_mask=self.dfg_to_dfg_mask[item]            
            self.tag[item]=True
            dfg_to_code=self.examples[item].dfg_to_code
            dfg_to_dfg=self.examples[item].dfg_to_dfg
            for i in range(len(dfg_to_code)):
                begin=min(dfg_to_code[i][0],self.args.max_source_length)
                end=min(dfg_to_code[i][1],self.args.max_source_length)
                dfg_to_code_mask[i,begin:end]=True
                for x in dfg_to_dfg[i]:
                    dfg_to_dfg_mask[i,x]=True
                dfg_to_dfg_mask[i,i]=True
                
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].target_ids),
                torch.tensor(self.examples[item].dfg_ids),
                torch.tensor(dfg_to_code_mask),
                torch.tensor(dfg_to_dfg_mask))






def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--tokenizer_name", default="", required=True,
                        help="Pretrained tokenizer name or path if not the same as model_name")    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_source_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--train_target_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_source_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_target_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_source_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_target_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_dfg_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--language', type=str, default='')    
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,do_lower_case=args.do_lower_case)
    
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)




    if args.do_train:
        # Prepare training data loader
        train_dataset =TextDataset(tokenizer, args, args.train_source_filename,args.train_target_filename)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
    
        
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps*args.train_batch_size//len(train_dataset))
        
                
        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids,target_ids,dfg_ids,dfg_to_code_mask,dfg_to_dfg_mask = batch
            loss,_,_ = model(source_ids,dfg_ids,dfg_to_code_mask,dfg_to_dfg_mask,target_ids=target_ids)
            
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                #Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True
                
            if args.do_eval and ((global_step + 1) %args.eval_steps == 0) and eval_flag:
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_dataset=dev_dataset['dev_loss']
                else:
                    eval_dataset =TextDataset(tokenizer, args, args.dev_source_filename,args.dev_target_filename) 
                    dev_dataset['dev_loss']=eval_dataset
                eval_sampler = SequentialSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
                
                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_dataset))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,target_ids,dfg_ids,dfg_to_code_mask,dfg_to_dfg_mask = batch              

                    with torch.no_grad():
                        _,loss,num = model(source_ids,dfg_ids,dfg_to_code_mask,dfg_to_dfg_mask,target_ids=target_ids)     
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   
                
                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  
                            
                            
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_dataset=dev_dataset['dev_bleu']
                else:
                    eval_dataset =TextDataset(tokenizer, args, args.dev_source_filename,args.dev_target_filename) 
                    eval_dataset.examples=random.sample(eval_dataset.examples,min(1000,len(eval_dataset)))  
                    dev_dataset['dev_bleu']=eval_dataset


                
                eval_sampler = SequentialSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,target_ids,dfg_ids,dfg_to_code_mask,dfg_to_dfg_mask = batch                
                    with torch.no_grad():
                        preds = model(source_ids,dfg_ids,dfg_to_code_mask,dfg_to_dfg_mask)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for idx,(ref,gold) in enumerate(zip(p,eval_dataset.examples)):
                        predictions.append(ref)
                        f.write(ref+'\n')
                        f1.write(gold.target+'\n')       

                dev_bleu = round(_bleu(os.path.join(args.output_dir,"dev.gold"), 
                                 os.path.join(args.output_dir,"dev.output")),2)   
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)    
                if dev_bleu>best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               
    if args.do_test:
        files=[]
        if args.dev_source_filename is not None:
            files.append((args.dev_source_filename,args.dev_target_filename))
        if args.test_target_filename is not None:
            files.append((args.test_source_filename,args.test_target_filename))
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            eval_dataset =TextDataset(tokenizer, args, file[0],file[1]) 

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval() 
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)            
                source_ids,target_ids,dfg_ids,dfg_to_code_mask,dfg_to_dfg_mask = batch                
                with torch.no_grad():
                    preds = model(source_ids,dfg_ids,dfg_to_code_mask,dfg_to_dfg_mask) 
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for i,(ref,gold) in enumerate(zip(p,eval_dataset.examples)):
                    predictions.append(ref)
                    f.write(ref+'\n')
                    f1.write(gold.target+'\n')     

            dev_bleu = round(_bleu(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))), 
                                 os.path.join(args.output_dir,"test_{}.output".format(str(idx)))),2)   
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  "+"*"*20)    



                          

                
                
if __name__ == "__main__":
    main()


