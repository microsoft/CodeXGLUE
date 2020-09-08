# CodeXGLUE -- Text2Code Generation

Here are the dataset and pipeline for text-to-code generation task.

## Task Definition

Generate source code of class member functions in Java, given natural language description and class environment. Class environment is the programmatic context provided by the rest of the class, including other member variables and member functions in class. Models are evaluated by exact match and BLEU.

It's a challenging task because the desired code can vary greatly depending on the functionality the class provides. Models must (a) have a deep understanding of NL description and map the NL to environment variables, library API calls and user-defined methods in the class, and (b) decide on the structure of the resulting code.


## Dataset

### Concode dataset
We use concode dataset which is a widely used code generation dataset from Iyer's EMNLP 2018 paper [Mapping Language to Code in Programmatic Context](https://www.aclweb.org/anthology/D18-1192.pdf).

We have downloaded his published dataset and followed his preprocessed script. You can find the preprocessed data in `dataset/concode` directory.

Data statistics of concode dataset are shown in the below table:

|         |  #Examples  |
| ------- | :---------: |
|  Train  |   100,000   |
|   Dev   |    2,000    |
|  Test   |    2,000    |

### Data Format

Code corpus are saved in json lines format files. one line is a json object:
```
{
  "nl": "Increment this vector in this place. con_elem_sep double[] vecElement con_elem_sep double[] weights con_func_sep void add(double)",
  "code": "public void inc ( ) { this . add ( 1 ) ; }"
}
```

`nl` combines natural language description and class environment. Elements in class environment are seperated by special tokens like `con_elem_sep` and `con_func_sep`.

## Evaluator

We provide a script to evaluate predictions for this task, and report exact match and BLEU score. You can run the script like this:

```bash
python evaluator/evaluator.py -a=evaluator/answers.json -p=evaluator/predictions.txt
```

The outputs are:
```
BLEU: 20.21, EM: 17.0
```

### Input Format

Answer file is in the same format of the dev set json lines file. A legal prediction file is expected to be a txt format file. It should have the same number of lines as answer file. Each line is the model prediction for the corresponding input in answer file. For example, one line in the answer file is:
```
{
  "nl": "Increment this vector in this place. con_elem_sep double[] vecElement con_elem_sep double[] weights con_func_sep void add(double)",
  "code": "public void inc ( ) { this . add ( 1 ) ; }"
}
```

And the corresponding line in your prediction file is:
```
public void inc ( ) { this . add ( 1 ) ; }
```


## Pipeline

We provide a pipeline that fine-tunes our pre-trained GPT-2 model on this task.

## Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0

### Fine-tune
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

It might take 15 hours for fine-tuning on 4 32G NVIDIA V100.

### Evaluation

It's recommanded to run evaluation on single GPU

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

It might take 40 minutes for inferencing on a single 16G NVIDIA P100.

## Result

The results on concode test set are shown as below:

| Model                                                 |   EM    |   BLEU   | CodeBLEU |
| ----------------------------------------------------- | :-----: | :------: | :------: |
| Seq2Seq                                               |  3.05   |  21.31   |   17.61  |
| Seq2Action+MAML (ACL 2019)                            |  10.05  |  24.40   |   20.99  |
| Iyer-Simp+200 idoms (EMNLP 2020)                      |  12.20  |  26.60   |     -    |
| Transformer w/ GPT-2 (12L)                            |  17.35  |  25.37   |   22.79  |
| Transformer w/ CodeGPT (12L)                          |  18.25  |  28.69   |   25.69  |
| Transformer w/ CodeGPT (domain adapt from GPT-2, 12L) |**20.10**|**32.79** | **27.74**|

