# CodeXGLUE -- Code Completion (token level)

Here is the pipeline for token level code completion task.

## Task Definition

Predict next code token given context of previous tokens. Models are evaluated by token level accuracy.

Code completion is a one of the most widely used features in software development through IDEs. An effective code completion tool could improve software developers' productivity. We provide code completion evaluation tasks in two granularities -- token level and line level. Here we introduce token level code completion. Token level task is analogous to language modeling. Models should have be able to predict the next token in arbitary types.


## Dataset

We collect and provide two datasets for code completion. One in python, the other in java.


### py150 dataset

We use py150 dataset from Raychev's OOPSLA 2016 paper [Probabilistic Model for Code with Decision Trees](https://files.sri.inf.ethz.ch/website/papers/oopsla16-dt.pdf).

To download and preprocess the dataset, navigate to `dataset/py150` directory, and run
```shell
bash download_and_extract.sh
python preprocess.py --base_dir=py150_files --output_dir=token_completion
```

### Github Java Corpus

We use java corpus dataset mined by Allamanis, in his MSR 2013 paper [Mining Source Code Repositories at Massive Scale using Language Modeling](https://homepages.inf.ed.ac.uk/csutton/publications/msr2013.pdf). We follow the same split and preprocessing in Karampatsis's ICSE 2020 paper [Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code](http://homepages.inf.ed.ac.uk/s1467463/documents/icse20-main-1325.pdf).

To download the preprocessed dataset, navigate to `dataset/javaCorpus` directory, and run
```shell
bash download.sh
python preprocess.py --base_dir=token_completion --output_dir=token_completion
```

### Data Format

Code corpus are saved in txt format files. one line is a tokenized code snippets:
```
<s> from django . utils . translation import ugettext_lazy as _ <EOL> import horizon <EOL> from openstack_dashboard . dashboards . project import dashboard <EOL> class Stacks ( horizon . Panel ) : <EOL> name = _ ( "Stacks" ) <EOL> slug = "stacks" <EOL> permissions = ( '' , ) <EOL> dashboard . Project . register ( Stacks ) </s>
```

We have added `<s>` and `</s>` to indicate the start and the end of one piece of code. `<EOL>` is also added in python corpus to mark end of a line since in python there is no `;` or `}` to mark the end of a statement like in java.


### Data Statistics

Data statistics of py150 dataset are shown in the below table, note that there doesn't exist dev set in the origin py150 dataset, we choose the first 5,000 files in test set as dev set.

| Data Split  |   #Files    |   #Tokens   |
| ----------- | :---------: | :---------: |
|    Train    |   100,000   |    76.3M    |
|     Dev     |    5,000    |     3.8M    |
|    Test     |    50,000   |    37.2M    |

Data statistics of Github Java Corpus dataset are shown in the below table:

| Data Split  |   #Files   |   #Tokens   |
| ----------- | :--------: | :---------: |
|    Train    |   12,934   |   15.74M    |
|     Dev     |    7,189   |    3.83M    |
|    Test     |    8,268   |    5.32M    |


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

And the corresponding line in your prediction file is:
```
. import numpy <EOL> json . dump ( open ) <EOL>
```
The accuracy on this line is 62.5%


## Pipeline

We provide a pipeline that fine-tunes our pre-trained GPT-2 model, which we called CodeGPT, on this task.

CodeGPT is a "dessert" GPT-2 model which is pre-trained on CodeSearchNet dataset w/o OpenAI GPT-2 initializing. So it has its own vocabulary on code. We provide two versions of CodeGPT, one is on java, the other is on python. You can easily load them by huggingface transformers.

### Dependency

- python 3.6 or 3.7
- torch>=1.4.0
- transformers>=2.5.0
- fuzzywuzzy

### Fine-tune
To fine-tune CodeGPT on javaCorpus dataset for code completion in multi-GPU on a single machine, navigate to `code` directory, run:

```shell
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=microsoft/CodeGPT-small-java        # microsoft/CodeGPT-small-py for py150
LOGFILE=completion_javaCorpus.log
PER_NODE_GPU=YOUR_GPU_NUM       # modify YOUR_GPU_NUM

python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_lm.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=5 \            # We set 20 epochs for py150
        --logging_steps=100 \
        --save_steps=1000 \
        --overwrite_output_dir \
        --seed=42 \
        --not_pretrain
```

We stop at 50000 steps on py150 experiment, which takes 25 hours. And 2 hours with 2000 steps on java dataset. Both experiments run on 2 NVIDIA P100.

### Evaluation && Inference

It's recommanded to run evaluation on single GPU. The predictions will be saved at `$OUTPUTDIR/predictions.txt`

```shell
export CUDA_VISIBLE_DEVICES=0
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../save/javaCorpus/checkpoint       # directory of your saved model
LOGFILE=completion_javaCorpus_eval.log

python -u run_lm.py \
        --data_dir=$DATADIR \
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
| LSTM (Kim, 2020)                                      |    58.0    |
| Transformer (Facebook, 6L) (Kim, 2020)                |    68.1    |
| Transformer (12L)                                     |    73.26   |
| Transformer w/ GPT-2 (12L)                            |    74.22   |
| Transformer w/ CodeGPT (12L)                          |  **74.93** |

### javaCorpus

| Model                                                 |  Accuracy  |
| ----------------------------------------------------- | :--------: |
| BPE+LSTM (ICSE 2020*)                                 |    56.02   |
| Transformer (12L)                                     |    64.16   |
| Transformer w/ GPT-2 (12L)                            |    74.89   |
| Transformer w/ CodeGPT (12L)                          |  **76.45** |

\* We reproduced his experiment since this paper only reported MRR on the first 1,000,000 tokens in test set.

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
