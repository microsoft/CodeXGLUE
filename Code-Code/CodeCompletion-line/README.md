# CodeXGLUE -- Code Completion (line level)

Here is the introduction and pipeline for line level code completion task.

## Task Definition

Complete the unfinished line given previous context. Models are evaluated by exact match and edit similarity.

We propose line completion task to test model's ability to autocomplete a line. Majority code completion systems behave well in token level completion, but fail in completing an unfinished line like a method call with specific parameters, a function signature, a loop condition, a variable definition and so on. When a software develop finish one or more tokens of the current line, the line level completion model is expected to generate the entire line of syntactically correct code.

## Dataset

Line level code completion task shares the train/dev dataset with token level completion. After training a model on CodeCompletion-token, you could directly use it to test on line-level completion. 


### py150 line completion test set

We create test set from py150 token level code comepltion test set. Since we intend to test model's ability to autocomplete an arbitrary line, we select the line to be predicted at random. To generate an example, we randomly cut a file as two parts. The former part is the input context, models are expected to generating the code sequence in the latter part until the first $<EOL>$ token (excluding $<EOL>$).


Test set is already at `dataset/py150/line_completion/test.json`.

### Github Java Corpus line completion test set

We create test set from Github Java Corpus token level code comepltion test set. In the same way as for Python, we randomly cut a file as two parts. The former part is the input context, outputs is the code sequence in the latter part until the first ; or \{ and \} token (including ; or \} token, but excluding \{ token).

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
|    Test    |    10,000   |          477.81          |          6.61             |

Data statistics of Github Java Corpus line completion test set are shown in the below table:

| Data Split |  #Examples  | Average tokens of inputs | Average tokens of outputs |
| ---------- | :---------: | :----------------------: | :-----------------------: |
|    Test    |    3,000    |          365.00          |          7.13             |

## Evaluator

We provide a script to evaluate predictions for this task, and report exact match score and edit similarity. You can run the script like this:

```bash
python evaluator/evaluator.py -a=evaluator/answers.json -p=evaluator/predictions.txt
```

The outputs are:
```
Edit sim: 43.8, EM: 0.0
```

**Note** that when evaluating, the normalized literals will be converted to the original format, e.g. <NUM_LIT:1> => 1, '<STR_LIT>' => ''

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

We provide a pipeline that evaluate line completion on [CodeGPT](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token#codegpt) model. You could directly use the model trained on token level code completion to test on line-level completion. 

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0 and < 4.0.0
- fuzzywuzzy

### Inference

It's recommanded to run inference on single GPU. The predictions will be saved at `$OUTPUTDIR/predictions_line.txt`

```shell
export CUDA_VISIBLE_DEVICES=0
LANG=java                       # set python for py150
DATADIR=../dataset/javaCorpus/line_completion
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=../save/javaCorpus
PRETRAINDIR=../../CodeCompletion-token/save/javaCorpus/checkpoint
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
        --eval_line \
        --logging_steps=100 \
        --seed=42 
```

It might take 45 minutes for inferencing on py150 dataset and 15 minutes on java Corpus on a single NVIDIA P100.

## Result

### py150

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
| LSTM+BPE                                              |    23.77   |       56.26       |
| Transformer                                           |    38.51   |       69.01       |
| [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)                               |    41.73   |       70.60       |
| CodeGPT                                               |    42.18   |       71.23       |
| CodeGPT-adapted                                       |  **42.37** |     **71.59**     |

### javaCorpus

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
| BPE+LSTM                                              |    12.97   |       42.10       |
| Transformer                                           |    17.00   |       50.23       |
| [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)                            |    27.50   |       60.36       |
| CodeGPT                                               |    28.23   |       61.81       |
| CodeGPT-adapted                                       |  **30.60** |     **63.45**     |


## Reference

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
