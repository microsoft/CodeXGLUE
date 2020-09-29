
# CodeXGLUE -- Text-to-Text

## Task Definition

Translate code documentation between human languages. Models are evaluated by BLEU score.

## Dataset

The dataset we use is crawled and filtered from [Microsoft Documentation](https://docs.microsoft.com/), whose document located at https://github.com/MicrosoftDocs/.  




### Data Statistics
The source language and target language are stored in two files, one line corresponding to one line.
Data statistics of Microsoft multi-lingual documentation dataset are shown in the below table:

| Language Pairs        |   #Train    |   #Dev      |  #Test       |
| --------------------- | :---------: | :---------: |  :---------: |
| Danish <-> English    |    43K      |    1K       |   1K         |
| Latvian <-> English   |    19K      |    1K       |   1K         |
| Norwegian <-> English |    44K      |    1K       |   1K         |
|  Chinese <-> English  |    50K      |    1K       |   1K         |


### Data Preprocess

Preprocess dataset (add language sign at the begin of source language and combine all language pairs):

```shell

cd data
python preprocessing.py

```

## Evaluator

We provide a script to evaluate predictions for this task, and report BLEU score

### Example

```bash
python evaluator/evaluator.py  evaluator/output.txt -p evaluator/gold.txt
```

{bleu-4:  67.75}


## Pipeline

We also provide a pipeline that use Transformer architecture or fine-tunes [XLM-Roberta](https://arxiv.org/pdf/1911.02116.pdf) (--using_pretrain_model) on this task. 

### Dependency

- python 3.6 or 3.7
- torch==1.6.0
- transformers>=2.5.0



### Fine-tune


```shell

lr=5e-5
batch_size=32
beam_size=5
source_length=256
target_length=256
data_dir=../data/processed
output_dir=saved_models/multi_model
train_file=$data_dir/train.all.src,$data_dir/train.all.tgt
dev_file=$data_dir/dev.all.src,$data_dir/dev.all.tgt
eval_steps=5000 
train_steps=50000 
pretrained_model=xlm-roberta-base #CodeBERT: path to CodeBERT. Roberta: roberta-base


CUDA_VISIBLE_DEVICES=0,1 python run.py \
--do_train \
--do_eval \
--using_pretrain_model \
--model_type roberta \
--model_name_or_path $pretrained_model \
--config_name xlm-roberta-base \
--tokenizer_name xlm-roberta-base \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--train_steps $train_steps \
--eval_steps $eval_steps

```


### Evaluation

```shell

beam_size=5
batch_size=32
source_length=256
target_length=256
output_dir=saved_models/multi_model
data_dir=../data/processed
dev_file=$data_dir/dev.all.src,$data_dir/dev.all.tgt
test_file=$data_dir/test.all.src,$data_dir/test.all.tgt
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

CUDA_VISIBLE_DEVICES=0,1 python run.py \
--do_test \
--model_type roberta \
--model_name_or_path xlm-roberta-base \
--config_name xlm-roberta-base \
--tokenizer_name xlm-roberta-base  \
--load_model_path $test_model \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size

    
```

## Result

The results on multi-lingual test dataset are shown as below:


| Translation Direction  | Transformer | Pretrained Transformer|   
| -----------            |   :-------: |  :--------: | 
| English -> Danish      |    53.31    |    67.09    |  
| English  -> Latvian    |    37.85    |   51.92     |   
| English -> Norwegian   |    53.84    |    68.00    |
| English -> Chinese     |    59.90    |   70.60     | 
| Danish -> English      |    58.73    |    67.02    |  
| Latvian -> English     |   50.37     |   68.30     |   
| Norwegian -> English   |    57.33    |    71.84    |
| Chinese -> English     |    50.00    |    64.47    | 

