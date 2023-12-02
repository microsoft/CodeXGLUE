# CodeXGLUE -- Code Completion (token level)

**Update 2021.07.30:** We update the code completion dataset with literals normalized to avoid sensitive information. 

Here is the introduction and pipeline for token level code completion task.

## Task Definition

Predict next code token given context of previous tokens. Models are evaluated by token level accuracy.

Code completion is a one of the most widely used features in software development through IDEs. An effective code completion tool could improve software developers' productivity. We provide code completion evaluation tasks in two granularities -- token level and line level. Here we introduce token level code completion. Token level task is analogous to language modeling. Models should have be able to predict the next token in arbitary types.


## Dataset

We collect and provide two datasets for code completion. One in python, the other in java.


### Dependency

- python 3.7
- javalang == 0.13.0


### py150 dataset

We use py150 dataset from Raychev's OOPSLA 2016 paper [Probabilistic Model for Code with Decision Trees](https://files.sri.inf.ethz.ch/website/papers/oopsla16-dt.pdf).

To download and preprocess the dataset, navigate to `dataset/py150` directory, and run
```shell
bash download_and_extract.sh
python preprocess.py --base_dir=py150_files --output_dir=token_completion
```

### Github Java Corpus

We use java corpus dataset mined by Allamanis and Sutton, in their MSR 2013 paper [Mining Source Code Repositories at Massive Scale using Language Modeling](https://homepages.inf.ed.ac.uk/csutton/publications/msr2013.pdf). We follow the same split and preprocessing in Karampatsis's ICSE 2020 paper [Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code](http://homepages.inf.ed.ac.uk/s1467463/documents/icse20-main-1325.pdf).

To download the preprocessed dataset, navigate to `dataset/javaCorpus` directory, and run
```shell
bash download.sh
python preprocess.py --base_dir=token_completion --output_dir=token_completion
```

### Data Preprocessing

- Tokenization is applied since we focus the token-level completion. 
- We normalize uncommon literals for better user experience. Developers sometimes leave their names, IP address, phone numbers in their codes, and we don't encourage models to focus on these string or numeric literals. So we normalized these literals by some special tokens. Considering that frequently used literals may contain useful information, e.g. "\_\_main\_\_" or "utf-8", we preserve the 200 most frequent string and 30 most frequent numeric literals. These literals will be normalized by tokens in `"<STR_LIT:utf-8>"` format, while uncommon literals are replaced by `<STR_LIT>` or `<NUM_LIT>`. 
- We add `<s>` and `</s>` to indicate the start and the end of one piece of code. 
- `<EOL>` is added in python corpus to mark the ending of a line since in python there is no `;` or `}` to mark the ending of a statement like in java.

### Data Format

Code corpus are saved in txt format files. one line is a tokenized code snippets:
```
<s> from __future__ import unicode_literals <EOL> from django . db import models , migrations <EOL> class Migration ( migrations . Migration ) : <EOL> dependencies = [ <EOL> ] <EOL> operations = [ <EOL> migrations . CreateModel ( <EOL> name = '<STR_LIT>' , <EOL> fields = [ <EOL> ( '<STR_LIT:id>' , models . AutoField ( verbose_name = '<STR_LIT>' , serialize = False , auto_created = True , primary_key = True ) ) , <EOL> ( '<STR_LIT:name>' , models . CharField ( help_text = b'<STR_LIT>' , max_length = <NUM_LIT> ) ) , <EOL> ( '<STR_LIT:image>' , models . ImageField ( help_text = b'<STR_LIT>' , null = True , upload_to = b'<STR_LIT>' , blank = True ) ) , <EOL> ] , <EOL> options = { <EOL> '<STR_LIT>' : ( '<STR_LIT:name>' , ) , <EOL> '<STR_LIT>' : '<STR_LIT>' , <EOL> } , <EOL> bases = ( models . Model , ) , <EOL> ) , <EOL> ] </s>
```


### Data Statistics

Data statistics of py150 dataset are shown in the below table, note that there doesn't exist dev set in the origin py150 dataset, we select 5,000 files in the original train set as dev set.

| Data Split  |   #Files    |   #Tokens   |
| ----------- | :---------: | :---------: |
|    Train    |    95,000   |    72.1M    |
|     Dev     |    5,000    |     4.4M    |
|    Test     |    50,000   |    37.3M    |

Data statistics of Github Java Corpus dataset are shown in the below table:

| Data Split  |   #Files   |   #Tokens   |
| ----------- | :--------: | :---------: |
|    Train    |   12,934   |    15.7M    |
|     Dev     |    7,176   |     3.8M    |
|    Test     |    8,268   |     5.3M    |


## Evaluator

We provide a script to evaluate predictions for this task, and report accuracy score. You can run the script like this:

```bash
python evaluator/evaluator.py -a=evaluator/answers.txt -p=evaluator/predictions.txt
```

The outputs are:
```
Total 5315204 tokens, accuracy: 76.45
```

### Input Format

Answer file is in the same format of the preprocessed dev dataset file. A legal prediction file is expected to be a txt format file. It should have the same number of lines as answer file. And for each line, it should contain the same number of tokens (split by space) as the corresponding line in the answer file. Note that `<s>`, `</s>`, `<EOL>` are not evaluated so that you don't need worry about how to predict the first token. You can put any token you like at first. For example, one line in the answer file is:
```
<s> import json <EOL> json . load ( f ) </s>
```

And the corresponding line in your prediction file may be:
```
. import numpy <EOL> json . dump ( open ) <EOL>
```
The accuracy on this line is 62.5%


## Pipeline


### CodeGPT

we provide CodeGPT, which is a Transformer-based language model pre-trained on programming language (PL). CodeGPT shares the same model architecture and training object with GPT-2, consisting 12 layers of Transformer decoders. We pre-train monolingual models respectively on Python and Java corpus from the CodeSearchNet dataset, which includes 1.1M Python functions and 1.6M Java methods. A function or method in training dataset consists function signature and function body. Some functions also contain NL docstrings. The dataset statistics are shown below:
|            | #Functions |   #Tokens   |
| ---------- | :--------: | :---------: |
|   Python   | 1,144,977  |   119.0M    |
|    Java    | 1,554,613  |   169.4M    |

We release two CodeGPT models for each programming language. One model is pre-trained from scratch, in a way that the BPE (byte pair encoder) vocabulary is newly obtained on code corpus and that model parameters are randomly initialized. The other model is a domain-adaptive one, which uses GPT-2 model as the starting point and is continually trained on code corpus. Therefore, the second model has the same vocabulary with GPT-2, and inherits the natural language understanding ability of GPT-2. It might perform better on natural language related tasks. We call the second model CodeGPT-adapted and regard it as the default one. 

All the models are publicly available at [huggingface website](https://huggingface.co/models?search=microsoft). Model names are `CodeGPT-small-py`, `CodeGPT-small-java`, `CodeGPT-small-py-adaptedGPT2`, `CodeGPT-small-java-adaptedGPT2`


### Dependency

- python 3.6 or 3.7
- torch>=1.4.0
- transformers>=2.5.0 and < 4.0.0
- fuzzywuzzy

### Fine-tune
To fine-tune CodeGPT on javaCorpus dataset for code completion in multi-GPU on a single machine, navigate to `code` directory, run:

```shell
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=microsoft/CodeGPT-small-java        # microsoft/CodeGPT-small-py for py150
LOGFILE=completion_javaCorpus.log
PER_NODE_GPU=YOUR_GPU_NUM       # modify YOUR_GPU_NUM

python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_train \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=5 \
        --logging_steps=100 \
        --save_steps=1000 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain
```

We stop at 50000 steps on py150 experiment, which takes 25 hours. And 2 hours with 2000 steps on java dataset. Both experiments run on 2 NVIDIA P100.

### Evaluation && Inference

It's recommanded to run evaluation on single GPU. The predictions will be saved at `$OUTPUTDIR/predictions.txt`

```shell
export CUDA_VISIBLE_DEVICES=0
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../save/javaCorpus/checkpoint       # directory of your saved model
LOGFILE=completion_javaCorpus_eval.log

python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_eval \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=100 \
        --seed=42 
```

It might take 60 minutes for inference on py150 dataset and 15 minutes on java Corpus on a single NVIDIA P100.


## Result

### py150

| Model                                                 |  Accuracy  |
| ----------------------------------------------------- | :--------: |
| LSTM + BPE                                            |    61.94   |
| Transformer (12L)                                     |    74.48   |
| [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)                            |    75.90   |
| CodeGPT                                               |    76.58   |
| CodeGPT-adapted                                       |  **76.60** |

### javaCorpus

| Model                                                 |  Accuracy  |
| ----------------------------------------------------- | :--------: |
| LSTM + BPE                                            |    58.92   |
| Transformer (12L)                                     |    65.18   |
| [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)                           |    75.40   |
| CodeGPT                                               |    76.79   |
| CodeGPT-adapted                                       |  **77.73** |


## Reference

If you use code completion datasets, please also cite the following papers in addition to our CodeXGLUE:

<pre><code>@article{raychev2016probabilistic,
  title={Probabilistic Model for Code with Decision Trees},
  author={Raychev, Veselin and Bielik, Pavol and Vechev, Martin},
  journal={ACM SIGPLAN Notices},
  pages={731--747},
  year={2016},
  publisher={ACM New York, NY, USA}
}</code></pre>

<pre><code>@inproceedings{allamanis2013mining,
  title={Mining Source Code Repositories at Massive Scale using Language Modeling},
  author={Allamanis, Miltiadis and Sutton, Charles},
  booktitle={2013 10th Working Conference on Mining Software Repositories (MSR)},
  pages={207--216},
  year={2013},
  organization={IEEE}
}</code></pre>
