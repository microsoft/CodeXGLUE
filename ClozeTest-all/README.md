# CodeXGLUE -- ClozeTest-all

We present two kinds of ClozeTest: ClozeTest-maxmin and ClozeTest-all. Here is the pipeline for ClozeTest-all task.

## Task Description

Cloze tests are widely adopted in natural languages to evaluate models' understanding of language, which can be formulated as a multi-choice classification problem. 

Here we present the two cloze test datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word. 

The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.1.0


## Data

The ClozeTest data are collected from the the validation and test sets of CodeSearchNet, which contains six different programming languages including RUBY, JAVASCRIPT, GO, PYTHON, JAVA and PHP. Each instance contains a masked code function, its docstring and the target word. 

We present the preprocessed data in `data/cloze-all` directory. And the selected words for ClozeTest is in `data/cloze-all/cloze_test_words.txt`. 

Data statistics of ClozeTest-all are shown in the below table:

| RUBY | JAVASCRIPT |  GO   | PYTHON | JAVA  |  PHP  |  ALL   |
| :--: | :--------: | :---: | :----: | :---: | :---: | :----: |
| 4437 |   13837    | 25282 | 40137  | 40492 | 51930 | 176115 |


## Run ClozeTest

You can run ClozeTest-all by the following command:

```shell
python run_cloze.py \
			--model microsoft/codebert-base-mlm \
			--cloze_mode all
```

## Result

The results on ClozeTest-all are shown as below:

|               | RUBY  | JAVASCRIPT |  GO   | PYTHON | JAVA  |  PHP  |  ALL  |
| :-----------: | :---: | :--------: | :---: | :----: | :---: | :---: | :---: |
| RoBERTa-base  | 47.64 |   59.97    | 40.98 | 54.49  | 50.75 | 60.38 | 53.69 |
| CodeBERT(MLM) | 80.17 |   81.77    | 83.31 | 87.21  | 80.63 | 85.05 | 83.89 |

