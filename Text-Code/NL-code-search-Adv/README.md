# CodeXGLUE -- Code Search (AdvTest)

## Task Definition

Given a natural language, the task is to search source code that matches the natural language. To test the generalization ability of a model,  function names and variables in test sets are replaced by special tokens.

## Dataset

The dataset we use comes from [CodeSearchNet](https://arxiv.org/pdf/1909.09436.pdf) and we filter the dataset as the following:

- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.

Besides, to test the generalization ability of a model,  function names and variables in test sets are replaced by special tokens.

### Data Download ans Preprocess

```shell
unzip dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip
python preprocess.py
rm -r python
rm -r *.pkl
rm python.zip
cd ..
```

### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

  - **repo:** the owner/repo
  - **path:** the full path to the original file
  - **func_name:** the function or method name
  - **original_string:** the raw string before tokenization or parsing
  - **language:** the programming language
  - **code/function:** the part of the `original_string` that is code
  - **code_tokens/function_tokens:** tokenized version of `code`
  - **docstring:** the top-level comment or docstring, if it exists in the original string
  - **docstring_tokens:** tokenized version of `docstring`
  - **url:** the url for the example (identify natural language)
  - **idx**: the index of code (identify code)

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  251,820  |
| Dev   |   9,604   |
| Test  |  19,210   |

## Evaluator

We provide a script to evaluate predictions for this task, and report MRR score.

### Example

Given a text-code file evaluator/test.jsonl:

```json
{"url": "url0", "docstring": "doc0","function": "fun0", "idx": 10}
{"url": "url1", "docstring": "doc1","function": "fun1", "idx": 11}
{"url": "url2", "docstring": "doc2","function": "fun2", "idx": 12}
{"url": "url3", "docstring": "doc3","function": "fun3", "idx": 13}
{"url": "url4", "docstring": "doc4","function": "fun4", "idx": 14}
```

Report MRR score

```shell
python evaluator/evaluator.py -a evaluator/test.jsonl  -p evaluator/predictions.jsonl 
```

{'MRR': 0.4233}

### Input Predictions

For each url for natural language, descending sort candidate codes and return their idx in order. For example:

```json
{"url": "url0", "answers": [10,11,12,13,14]}
{"url": "url1", "answers": [10,12,11,13,14]}
{"url": "url2", "answers": [13,11,12,10,14]}
{"url": "url3", "answers": [10,14,12,13,11]}
{"url": "url4", "answers": [10,11,12,13,14]}
```

## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task. 

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0


### Fine-tune

To fine-tune encoder-decoder on the dataset

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 2 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
```


### Inference

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 2 \
    --block_size 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee test.log
```

### Evaluation

```shell
python ../evaluator/evaluator.py -a ../dataset/test.jsonl  -p saved_models/predictions.jsonl 
```

{'MRR': 0.2719}

## Result

The results on the test set are shown as below:

| Method   |     MRR    |
| -------- | :--------: |
| [Roberta](https://arxiv.org/pdf/1907.11692.pdf)  |   0.1833   |
| [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) | **0.2719** |

## Reference
<pre><code>@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}</code></pre>
