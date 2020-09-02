# CodeXGLUE -- ClozeTest-maxmin

We present two kinds of ClozeTest: ClozeTest-maxmin and ClozeTest-all. Here is the pipeline for ClozeTest-maxmin task.

## Task Description

Cloze tests are widely adopted in natural languages to evaluate models' understanding of language, which can be formulated as a multi-choice classification problem. 

Here we present the two cloze test datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word. 

The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.1.0


## Data

The ClozeTest data are collected from the the validation and test sets of CodeSearchNet, which contains six different programming languages including RUBY, JAVASCRIPT, GO, PYTHON, JAVA and PHP. Each instance contains a masked code function, its docstring and the target word. An example in PYTHON is shown below:

We present the preprocessed data in `data/cloze-maxmin` directory. 

Data statistics of ClozeTest-maxmin are shown in the below table:

| RUBY | JAVASCRIPT |  GO  | PYTHON | JAVA | PHP  | ALL  |
| :--: | :--------: | :--: | :----: | :--: | :--: | :--: |
|  38  |    272     | 152  |  1264  | 482  | 407  | 2615 |


## Run ClozeTest

You can run ClozeTest-maxmin by the following command:

```shell
python run_cloze.py \
			--model microsoft/codebert-base-mlm \
			--cloze_mode maxmin
```

## Result

The results on ClozeTest-maxmin are shown as below:

|               | RUBY  | JAVASCRIPT |  GO   | PYTHON | JAVA  |  PHP  |  ALL  |
| :-----------: | :---: | :--------: | :---: | :----: | :---: | :---: | :---: |
| RoBERTa-base  | 47.64 |   59.97    | 40.98 | 54.49  | 50.75 | 60.38 | 53.69 |
| CodeBERT(MLM) | 80.17 |   81.77    | 83.31 | 87.21  | 80.63 | 85.05 | 83.89 |

