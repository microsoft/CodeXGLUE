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

We provide a pipeline that fine-tunes our pre-trained GPT-2 model, which we called CodeGPT, on this task.

CodeGPT is a "dessert" GPT-2 model which is pre-trained on Python and Java dataset (PL data only) from CodeSearchNet w/o OpenAI GPT-2 initializing. Below are the statistics for training datasets.
|            | #Functions |   #Tokens   |
| ---------- | :--------: | :---------: |
|   Python   | 1,144,977  |   119.0M    |
|    Java    | 1,554,613  |   169.4M    |

We provide two versions of CodeGPT, one is for Java, the other is for Python. Each of them has its own vocabulary on code. You can easily load them by huggingface transformers.

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
PRETRAINDIR=microsoft/CodeGPT-small-java    # will download pre-trained CodeGPT model
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
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=6 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=30 \
        --logging_steps=100 \
        --save_steps=5000 \
        --overwrite_output_dir \
        --seed=42
```

We stop at 60000 steps, which takes 22 hours on 2 NVIDIA P100.

### Evaluation

It's recommanded to run evaluation on dev set on single GPU. The predictions on dev set will be saved in `$OUTPUTDIR/dev.output`.

```shell
export CUDA_VISIBLE_DEVICES=0
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

### Inference

It's recommanded to run inference on test set on single GPU. The predictions will be saved in `$OUTPUTDIR/test.output`.

```shell
export CUDA_VISIBLE_DEVICES=0
LANG=java
DATADIR=../dataset/concode
OUTPUTDIR=../save/concode
PRETRAINDIR=../save/concode/checkpoint
LOGFILE=text2code_concode_infer.log

python -u run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_infer \
        --logging_steps=100 \
        --seed=42
```

It might take 40 minutes for inference on a single NVIDIA P100.

## Result

The results on concode test set are shown as below:

| Model                                                 |   EM    |   BLEU   | CodeBLEU |
| ----------------------------------------------------- | :-----: | :------: | :------: |
| Seq2Seq                                               |  3.05   |  21.31   |   17.61  |
| Seq2Action+MAML (ACL 2019)                            |  10.05  |  24.40   |   20.99  |
| Iyer-Simp+200 idoms (EMNLP 2020)                      |  12.20  |  26.60   |     -    |
| Transformer w/ GPT-2 (12L)                            |  17.35  |  25.37   |   22.79  |
| Transformer w/ CodeGPT (12L)                          |**18.25**|**28.69** | **25.69**|

## Reference

If you use concode dataset, please also cite this paper in addition to our CodeXGLUE:

<pre><code>@article{iyer2018mapping,
  title={Mapping language to code in programmatic context},
  author={Iyer, Srinivasan and Konstas, Ioannis and Cheung, Alvin and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1808.09588},
  year={2018}
}</code></pre>

