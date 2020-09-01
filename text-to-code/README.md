# CodeXGLUE -- Text2Code Generation

Here is the pipeline for text-to-code generation task.


## Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0


## Data Preprocess

### Concode dataset
We use concode dataset which is a widely used code generation dataset from Iyer's EMNLP 2018 paper [Mapping Language to Code in Programmatic Context](https://www.aclweb.org/anthology/D18-1192.pdf).

We have downloaded his published dataset and followed his preprocessed script. You'll find the preprocessed data in `dataset/concode` directory.

Data statistics of concode dataset are shown in the below table:

|         |  #Examples  |
| ------- | :---------: |
|  Train  |   100,000   |
|   Dev   |    2,000    |
|  Test   |    2,000    |


## Fine-tune
To fine-tune CodeGPT on concode dataset for text2code generation on multi-GPUs at a single machine, navigate to `code` directory, run:

```shell
LANG=java
DATADIR=../dataset/concode
OUTPUTDIR=../save/concode
PRETRAINDIR=../pretrained/CodeGPT/java/checkpoint
LOGFILE=text2code_concode.log
PER_NODE_GPU=YOUR_GPU_NUM       # modify YOUR_GPU_NUM

python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=4e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=8 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=1 \
        --num_train_epochs=50 \
        --logging_steps=200 \
        --save_steps=1000 \
        --overwrite_output_dir \
        --seed=42
```


## Evaluation

It's recommanded to run evaluation on single GPU

### Token level completion
```shell
LANG=java
DATADIR=../dataset/concode
OUTPUTDIR=../save/concode
PRETRAINDIR=../save/concode/checkpoint
LOGFILE=text2code_concode_eval.log

python -u run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_eval \
        --logging_steps=100 \
        --seed=42
```

## Result

The results on concode test set are shown as below:

| Model                                                 |   EM    |   BLEU   |  CodeBLEU  |
| ----------------------------------------------------- | :-----: | :------: | :--------: |
| Seq2Seq                                               |  3.05   |  21.31   |   17.61    |
| Seq2Action+MAML (ACL 2019)                            |  10.05  |  24.40   |   20.99    |
| Iyer-Simp+200 idoms (EMNLP 2020)                      |  12.20  |  26.60   |     /      |
| Transformer w/ GPT-2 (12L)                            |  17.35  |  25.37   |   22.79    |
| Transformer w/ CodeGPT (12L)                          |  18.25  |  28.61   |   25.69    |
| Transformer w/ CodeGPT (domain adapt from GPT-2, 12L) |**20.10**|**32.79** | **27.74**  |

