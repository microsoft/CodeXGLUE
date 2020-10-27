# CodeXGLUE -- Defect Detection

## Task Definition

Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack.  We treat the task as binary classification (0/1), where 1 stands for insecure code and 0 for secure code.

### Dataset

The dataset we use comes from the paper [*Devign*: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks](http://papers.nips.cc/paper/9209-devign-effective-vulnerability-identification-by-learning-comprehensive-program-semantics-via-graph-neural-networks.pdf). We combine all projects and split 80%/10%/10% for training/dev/test.

### Download and Preprocess

1.Download dataset from [website](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view?usp=sharing) to "dataset" folder or run the following command:

```shell
cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
cd ..
```

2.Preprocess dataset

```shell
cd dataset
python preprocess.py
cd ..
```

### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the source code
   - **target:** 0 or 1 (vulnerability or not)
   - **idx:** the index of example

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  21,854   |
| Dev   |   2,732   |
| Test  |   2,732   |

## Evaluator

We provide a script to evaluate predictions for this task, and report accuracy score.

### Example

```shell
python evaluator/evaluator.py -a evaluator/test.jsonl -p evaluator/predictions.txt
```

{'Acc': 0.6}

### Input predictions

A predications file that has predictions in TXT format, such as evaluator/predictions.txt. For example:

```shell
0	0
1	1
2	1
3	0
4	0
```

## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task.

### Fine-tune

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```


### Inference

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log
```

### Evaluation

```shell
python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/predictions.txt
```

{'Acc': 0.6207906295754027}

## Result

The results on the test set are shown as below:

| Methods  |    ACC    |
| -------- | :-------: |
| BiLSTM   |   59.37   |
| TextCNN  |   60.69   |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)  |   61.05   |
| [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) | **62.08** |

## Reference
<pre><code>@inproceedings{zhou2019devign,
  title={Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks},
  author={Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10197--10207},
  year={2019}
}</code></pre>
