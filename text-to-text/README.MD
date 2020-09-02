# CodeXGLUE -- Text-to-Text

Here is the pipeline for text-to-text task.


## Dependency

- python 3.6 or 3.7
- torch==1.6.0
- transformers>=2.5.0


## Data Preprocess

Data statistics of Microsoft multi-lingual documentation dataset are shown in the below table:

| Language Pairs        |   #Train    |   #Dev      |  #Test       |
| --------------------- | :---------: | :---------: |  :---------: |
| Danish <-> English    |    43K      |    1K       |   1K         |
| Latvian <-> English   |    19K      |    1K       |   1K         |
| Norwegian <-> English |    44K      |    1K       |   1K         |
|  Chinese <-> English  |    50K      |    1K       |   1K         |

Preprocess dataset (add language sign at the begin of source language and combine all language pairs):

```shell

cd data
python preprocessing.py

```


## Fine-tune

To fine-tune encoder-decoder (based on XLM-Roberta) on the dataset

```shell

lr=5e-5
batch_size=64
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


CUDA_VISIBLE_DEVICES=0,1,2,3 /home/espnet/tools/venv/bin/python run.py \
--do_train \
--do_eval \
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


## Evaluation

```shell

beam_size=5
batch_size=64
source_length=256
target_length=256
output_dir=saved_models/multi_model
data_dir=../data/processed
dev_file=$data_dir/dev.all.src,$data_dir/dev.all.tgt
test_file=$data_dir/test.all.src,$data_dir/test.all.tgt
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/espnet/tools/venv/bin/python run.py \
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


## Results

The results on the test set are shown as below:

| Direction              |   Dev     | Test       |  
| -----------            | :-------: | :--------: | 
| English  Danish        |   69.72   |   69.46    |  
| English  -> Latvian    |   54.38   |   55.92    |   
| English -> Norwegian   |   70.00   |   70.23    |
| English -> Chinese     |   73.89   |   73.54    | 
| Danish -> English      |   77.22   |   75.85    |  
| Latvian -> English     |   69.52   |   70.73    |   
| Norwegian -> English   |   75.18   |   74.63    |
| Chinese -> English     |   68.24   |   67.01    | 

