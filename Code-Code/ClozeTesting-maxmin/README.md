# CodeXGLUE -- ClozeTesting-maxmin

We present two kinds of cloze testing: ClozeTesting-maxmin and ClozeTesting-all. Here is the ClozeTesting-maxmin task.

## Task Description

Cloze testings are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem. 

Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTesting-maxmin and ClozeTesting-all. Each instance in the dataset contains a masked code function, its docstring and the target word. 

The only difference between ClozeTesting-maxmin and ClozeTesting-all is their selected words sets, where ClozeTesting-maxmin only contains two words while ClozeTesting-all contains 930 words.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.1.0


## Data

The data for cloze testing are collected from the the validation and test sets of CodeSearchNet, which contains six different programming languages including RUBY, JAVASCRIPT, GO, PYTHON, JAVA and PHP. Each instance contains a masked code function, its docstring and the target word. An example in PYTHON is shown below:

We present the preprocessed data in `data/cloze-maxmin` directory. 

Data statistics of ClozeTesting-maxmin are shown in the below table:

| RUBY | JAVASCRIPT |  GO  | PYTHON | JAVA | PHP  | ALL  |
| :--: | :--------: | :--: | :----: | :--: | :--: | :--: |
|  38  |    272     | 152  |  1264  | 482  | 407  | 2615 |


## Run ClozeTesting

You can run ClozeTesting-maxmin by the following command. It will automatically generate predictions to ` --output_dir`.

```shell
python code/run_cloze.py \
			--model microsoft/codebert-base-mlm \
			--cloze_mode maxmin \
			--output_dir evaluator/predictions/
```

## Evaluator

We provide a script to evaluate predictions for ClozeTesting-maxmin, and report accuracy for the task. You can run by the following command:

```shell
python evaluator/evaluator.py \
			--answers evaluator/answers \
			--predictions evaluator/predictions
```

## Result

The results on ClozeTesting-maxmin are shown as below:

|               | RUBY  | JAVASCRIPT |  GO   | PYTHON | JAVA  |  PHP  |  ALL  |
| :-----------: | :---: | :--------: | :---: | :----: | :---: | :---: | :---: |
| RoBERTa-base  | 73.68 |   65.81    | 73.68 | 59.18  | 59.75 | 69.78 | 62.68 |
| CodeBERT(MLM) | 86.84 |   84.93    | 92.76 | 81.25  | 91.70 | 89.93 | 85.66 |

## Cite

ClozeTesting-maxmin is built upon CodeSearchNet dataset. If you use this code or our ClozeTesting-maxmin dataset, please considering citing CodeXGLUE, CodeBERT and CodeSearchNet:	

<pre><code>@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}</code>
</pre>

<pre>
<code>@article{feng2020codebert,
  title={CodeBERT: A Pre-Trained Model for Programming and Natural Languages},
  author={Feng, Zhangyin and Guo, Daya and Tang, Duyu and Duan, Nan and Feng, Xiaocheng and Gong, Ming and Shou, Linjun and Qin, Bing and Liu, Ting and Jiang, Daxin and others},
  journal={arXiv preprint arXiv:2002.08155},
  year={2020}
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













