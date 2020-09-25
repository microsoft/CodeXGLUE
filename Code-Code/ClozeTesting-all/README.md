# CodeXGLUE -- ClozeTest-all

We present two kinds of cloze testing: ClozeTest-maxmin and ClozeTest-all. Here is the ClozeTest-all task.

## Task Description

Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem. 

Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word. 

The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.5.0


## Data

The data for cloze testing are collected from the the validation and test sets of CodeSearchNet, which contains six different programming languages including RUBY, JAVASCRIPT, GO, PYTHON, JAVA and PHP. Each instance contains a masked code function, its docstring and the target word. 

We present the preprocessed data in `data/cloze-all` directory. And the selected words for ClozeTest is in `data/cloze-all/cloze_test_words.txt`. 

Data statistics of ClozeTest-all are shown in the below table:

| RUBY | JAVASCRIPT |  GO   | PYTHON | JAVA  |  PHP  |  ALL   |
| :--: | :--------: | :---: | :----: | :---: | :---: | :----: |
| 4437 |   13837    | 25282 | 40137  | 40492 | 51930 | 176115 |


## Run ClozeTest

You can run ClozeTest-all by the following command. It will automatically generate predictions to ` --output_dir`.

```shell
python code/run_cloze.py \
			--model microsoft/codebert-base-mlm \
			--cloze_mode all \
			--output_dir evaluator/predictions/
```

## Evaluator

We provide a script to evaluate predictions for ClozeTest-all, and report accuracy for the task. You can run by the following command:

```shell
python evaluator/evaluator.py \
			--answers evaluator/answers \
			--predictions evaluator/predictions
```

## Result

The results on ClozeTest-all are shown as below:

|               | RUBY  | JAVASCRIPT |  GO   | PYTHON | JAVA  |  PHP  |  ALL  |
| :-----------: | :---: | :--------: | :---: | :----: | :---: | :---: | :---: |
| RoBERTa-base  | 47.44 |   59.96    | 40.77 | 54.35  | 50.73 | 60.16 | 53.55 |
| CodeBERT(MLM) | 80.17 |   81.77    | 83.31 | 87.21  | 80.63 | 85.05 | 83.89 |

## Cite

ClozeTest-all is built upon CodeSearchNet dataset. If you use this code or our ClozeTest-all dataset, please considering citing CodeXGLUE and CodeSearchNet:	

<pre><code>@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}</code>
</pre>
<pre>
<code>@article{husain2019codesearchnet,
  title={CodeSearchNet Challenge: Evaluating the State of Semantic Code Search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}</code> 
</pre>

