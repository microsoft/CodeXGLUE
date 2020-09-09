# CodeXGLUE -- Code Completion (line level)

Here is the pipeline for line level code completion task.

## Task Definition

Complete the unfinished line given previous context. Models are evaluated by exact match and edit similarity.

We propose line completion task to test model's ability to autocomplete a line. Majority code completion systems behave well in token level completion, but fail in completing an unfinished line like a method call with specific parameters, a function signature, a loop condition, a variable definition and so on. When a software develop finish one or more tokens of the current line, the line level completion model is expected to generate the entire line of syntactically correct code.

## Dataset

Line level code completion task shares the train/dev dataset with token level completion. After training a model on CodeCompletion-token, you could directly use it to test on line-level completion. 


### py150 line completion test set

We create test set from py150 token level code comepltion test set. We select a file (one line in test.txt) and randomly cut it as two part. The former part is inputs, while all the tokens until the first `<EOL>` token (excluding `<EOL>`) in the latter part is outputs.

Test set is already at `dataset/py150/line_completion/test.json`.

### Github Java Corpus line completion test set

We create test set from Github Java Corpus token level code comepltion test set. We select a file (one line in test.txt) and randomly cut it as two part. The former part is inputs, while all the tokens until the first `;` or `}` token (including `;` or `}`) in the latter part is outputs.

Test set is already at `dataset/javaCorpus/line_completion/test.json`.

### Data Format

Data is saved in json lines format files. Each line is a json object. To be consistent with token level code completion, codes have been tokenized. Here is an example of one line:
```
{
  "input": "<s> from __future__ import absolute_import , division , print_function <EOL> from . _ithreads import AlreadyQuit <EOL> class Quit ( object ) : <EOL>",
  "gt": "def __init__ ( self ) :"
}
```


### Data Statistics

Data statistics of py150 line completion test set are shown in the below table:

| Data Split |  #Examples  | Average tokens of inputs | Average tokens of outputs |
| ---------- | :---------: | :----------------------: | :-----------------------: |
|    Test    |    10,000   |          489.11          |          6.56             |

Data statistics of Github Java Corpus line completion test set are shown in the below table:

| Data Split |  #Examples  | Average tokens of inputs | Average tokens of outputs |
| ---------- | :---------: | :----------------------: | :-----------------------: |
|    Test    |    3,000    |          350.62          |          10.49            |

## Evaluator

We provide a script to evaluate predictions for this task, and report exact match score and edit similarity. You can run the script like this:

```bash
python evaluator/evaluator.py -a=evaluator/answers.json -p=evaluator/predictions.txt
```

The outputs are:
```
Edit sim: 71.05, EM: 39.0
```

### Input Format

A legal prediction file is expected to be a txt format file. It should have the same number of lines as answer file. Each line is the model prediction for the corresponding input in answer file. For example, one line in the answer file is:
```
{
  "input": "<s> from __future__ import absolute_import , division , print_function <EOL> from . _ithreads import AlreadyQuit <EOL> class Quit ( object ) : <EOL>",
  "gt": "def __init__ ( self ) :"
}
```

And the corresponding line in your prediction file is:
```
def __init__ ( self ) :
```


## Pipeline

We provide a pipeline that evaluate line completion on our fine-tuned GPT-2 model. You could directly use the model trained on token level code completion to test on line-level completion. 

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0
- fuzzywuzzy

### Evaluation

```shell
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/line_completion
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../../CodeCompletion-token/save/javaCorpus/checkpoint
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

It might take 45 minutes for inferencing on py150 dataset and 15 minutes on java Corpus on a single 16G NVIDIA P100.

## Result

### py150

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
| BPE+LSTM                                              |    17.93   |       50.05       |
| Transformer (12L)                                     |    36.80   |       67.66       |
| Transformer w/ GPT-2 (12L)                            |    38.96   |       69.29       |
| Transformer w/ CodeGPT (12L)                          |    39.37   |       70.02       |
| Transformer w/ CodeGPT (domain adapt from GPT-2, 12L) |  **40.48** |     **70.48**     |

### javaCorpus

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
| BPE+LSTM                                              |    10.30   |       41.55       |
| Transformer (12L)                                     |    15.33   |       50.39       |
| Transformer w/ GPT-2 (12L)                            |    24.30   |       60.70       |
| Transformer w/ CodeGPT (12L)                          |    25.30   |       61.54       |
| Transformer w/ CodeGPT (domain adapt from GPT-2, 12L) |  **26.43** |     **63.03**     |

