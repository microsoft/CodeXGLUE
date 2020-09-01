# CodeXGLUE -- Code Completion

Here is the pipeline for code completion task.


## Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0
- fuzzywuzzy


## Data Preprocess

### py150 dataset

We use py150 dataset from Raychev's OOPSLA 2016 paper [Probabilistic Model for Code with Decision Trees](https://files.sri.inf.ethz.ch/website/papers/oopsla16-dt.pdf).

To download and preprocess the dataset, navigate to `dataset/py150` directory, and run
```shell
bash download_and_extract.sh
python preprocess.py --base_dir=py150_files --output_dir=token_completion
```

Data statistics of py150 dataset are shown in the below table, note that there doesn't exist dev set in the origin py150 dataset, we choose the first 5,000 files in test set as dev set.

| Token Level |   #Files    |   #Tokens   |
| ----------- | :---------: | :---------: |
|    Train    |   100,000   |    76.3M    |
|     Dev     |    5,000    |     3.8M    |
|    Test     |    50,000   |    37.2M    |

| Line Level |  #Examples  | Average tokens of inputs | Average tokens of outputs |
| ---------- | :---------: | :----------------------: | :-----------------------: |
|    Test    |    10,000   |          489.11          |          6.56             |

### Github Java Corpus

We use java corpus dataset mined by Allamanis, in his MSR 2013 paper [Mining Source Code Repositories at Massive Scale using Language Modeling](https://homepages.inf.ed.ac.uk/csutton/publications/msr2013.pdf). We follow the same split and preprocessing in Karampatsis's ICSE 2020 paper [Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code](http://homepages.inf.ed.ac.uk/s1467463/documents/icse20-main-1325.pdf).

To download the preprocessed dataset, navigate to `dataset/javaCorpus` directory, and run
```shell
bash download.sh
```

Data statistics of Github Java Corpus dataset are shown in the below table:

| Token Level |   #Files   |   #Tokens   |
| ----------- | :--------: | :---------: |
|    Train    |   12,934   |   15.74M    |
|     Dev     |    7,189   |    3.83M    |
|    Test     |    8,268   |    5.33M    |

| Line Level |  #Examples  | Average tokens of inputs | Average tokens of outputs |
| ---------- | :---------: | :----------------------: | :-----------------------: |
|    Test    |    3,000    |          350.62          |          10.49            |


Both two datasets are used for two code completion tasks -- token level completion and line level completion. These two tasks share train/dev set but not test set. Test sets for line level completion are already in `dataset/py150/line_completion` and `dataset/javaCorpus/line_completion`.


## Fine-tune
To fine-tune CodeGPT on javaCorpus dataset for code completion in multi-GPU on a single machine, navigate to `code` directory, run:

```shell
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../pretrained/CodeGPT/java/checkpoint
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
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=50 \
        --logging_steps=100 \
        --save_steps=1000 \
        --overwrite_output_dir \
        --seed=42 \
        --not_pretrain
```


## Evaluation

It's recommanded to run evaluation on single GPU

### Token level completion
```shell
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/token_completion
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../save/javaCorpus/checkpoint
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

### Line level completion
```shell
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/line_completion
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../save/javaCorpus/checkpoint
LOGFILE=completion_javaCorpus_eval.log

python -u run_lm.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 
```

## Result

### Token level completion

#### py150

| Model                                                 |  Accuracy  |
| ----------------------------------------------------- | :--------: |
| LSTM (Kim, 2020)                                      |    58.0    |
| Transformer (Facebook, 6L) (Kim, 2020)                |    68.1    |
| Transformer (12L)                                     |    73.35   |
| Transformer w/ GPT-2 (12L)                            |    74.51   |
| Transformer w/ CodeGPT (12L)                          |    75.09   |
| Transformer w/ CodeGPT (domain adapt from GPT-2, 12L) |  **75.28** |

#### javaCorpus

| Model                                                 |  Accuracy  |
| ----------------------------------------------------- | :--------: |
| BPE+LSTM (ICSE 2020*)                                 |    55.91   |
| Transformer (12L)                                     |    64.01   |
| Transformer w/ GPT-2 (12L)                            |    74.72   |
| Transformer w/ CodeGPT (12L)                          |    76.29   |
| Transformer w/ CodeGPT (domain adapt from GPT-2, 12L) |  **77.13** |

\* We reproduced his experiment since this paper only reported MRR on the first 1,000,000 tokens in test set.


### Line level completion

#### py150

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
| BPE+LSTM                                              |    17.93   |       50.05       |
| Transformer (12L)                                     |    36.80   |       67.66       |
| Transformer w/ GPT-2 (12L)                            |    38.96   |       69.29       |
| Transformer w/ CodeGPT (12L)                          |    39.37   |       70.02       |
| Transformer w/ CodeGPT (domain adapt from GPT-2, 12L) |  **40.48** |     **70.48**     |

#### javaCorpus

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
| BPE+LSTM                                              |    10.30   |       41.55       |
| Transformer (12L)                                     |    15.33   |       50.39       |
| Transformer w/ GPT-2 (12L)                            |    24.30   |       60.70       |
| Transformer w/ CodeGPT (12L)                          |    25.30   |       61.54       |
| Transformer w/ CodeGPT (domain adapt from GPT-2, 12L) |  **26.43** |     **63.03**     |

