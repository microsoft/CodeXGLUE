# This repo is under construction...

# Introduction
This repo includes datasets and codes for CodeXGLUE, a collection of code intelligence tasks and a platform for model evaluation and comparison. CodeXGLUE stands for General Language Understanding Evaluation benchmark for CODE. It includes 14 datasets for 10 diversified programming language tasks covering code-code (clone detection, defect detection, cloze test, code completion, code refinement, and code-to-code translation), text-code (natural language code search, text-to-code generation), code-text (code summarization) and text-text (documentation translation) scenarios. We provide three baseline models to support these tasks, including BERT-style pre-trained model (i.e. [CodeBERT](https://github.com/microsoft/CodeBERT)) which is good at understanding problems, GPT-style pre-trained model which we call CodeGPT to support completion and generation problems, and Encoder-Decoder framework that supports sequence-to-sequence generation problems.

A brief summary of CodeXGLUE is given below, including tasks, datasets, baseline systems, etc. Datasets highlighed in BLUE are fully contributed or partially contributed by Microsoft.
![A brief summary of CodeXGLUE, including tasks, datasets, baseline systems, etc. Datasets highlighed in BLUE are fully contributed or partially contributed by Microsoft.](https://github.com/microsoft/CodeXGLUE/blob/main/CodeXGLUE-table.jpg)

# Relevant Links
[Leaderboard](to-be-added) | [CodeXGLUE paper](to-be-added)

# Tasks and Datasets

## Clone Detection (Daya)
We have two datasets to detect the semantic equivalence and similairty between codes. 
The first dataset is ... Given xxx and xxx as input, the task is to predict. .. Models are evaluated by ...
The second dataset is ... . Given a piece of code as the input, the task is to ... Models are evaluated by ...


## Defect Detection (Daya)


## Cloze Test (junjie)

## Code Completion (Shuai)
We have both token-level and line-level completion tasks. ...

## Code Refinement (Shuo)

## Code Translation (Shuo)

## Natural Language Code Search (Daya & Junjie)
We have two datasets to detect semantic similarity between text in natural language and code in programming langauge. 

The first dataset is (Daya)

The second dataset is (Junjie)

## Text-to-Code Generation (Shuai)

## Code Summarization (Daya)

## Documentation Translation (Long)

# CodeXGLUE Submission Instructions
Once you have built a model that meets your expectations on evaluation with the dev set, you can submit your test results to get official evaluation on the test set. To ensure the integrity of the official test results, we do not release the correct answers for test set to the public. To submit your model for official evaluation on the test set, follow the below steps:
1. Generate your prediction output for the dev set.
2. Run the official evaluation methodologies found in the task specific git repo and verify your systems are running as expected.
3. Generate your prediction output for the test set and submit the following information by emailing us.

Your email should include:

* Prediction results on test set. [Required]
* Prediction results on dev set. [Recommended]
* Individual/Team Name: Name of the individual or the team to appear in the leaderboard. [Required]
* Individual/Team Institution: Name of the institution of the individual or the team to appear in the leaderboard. [Optional]
* Model code: Training code for the model. [Recommended]
* Model information: Name of the model/technique to appear in the leaderboard. [Required]
* Paper Information: Name, Citation, URL of the paper if model is from a published work to appear in the leaderboard. [Optional]

To avoid "P-hacking" we discourage too many submissions from the same group in a short period of time.



# How to Cite
<pre><code>@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}</code></pre>

CodeXGLUE is built out of the following datasets. Please ensure you cite all of them.

<pre><code>@article{husain2019codesearchnet,
title={CodeSearchNet Challenge: Evaluating the State of Semantic Code Search},
author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
journal={arXiv preprint arXiv:1909.09436},
year={2019}
}</code></pre>

<pre><code>@article{DBLP:journals/corr/abs-1812-08693,
  title= {An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation},
  author= {Michele Tufano, Cody Watson, Gabriele Bavota, Massimiliano Di Penta, Martin White and Denys Poshyvanyk},
  journal= {arXiv abs/1812.08693},
  yea= {2018}
}</code></pre>

<pre><code>@article{raychev2016probabilistic,
title={Probabilistic model for code with decision trees},
author={Raychev, Veselin and Bielik, Pavol and Vechev, Martin},
journal={ACM SIGPLAN Notices},
volume={51},
number={10},
pages={731--747},
year={2016},
publisher={ACM New York, NY, USA}
}</code></pre>

<pre><code>@inproceedings{allamanis2013mining,
title={Mining source code repositories at massive scale using language modeling},
author={Allamanis, Miltiadis and Sutton, Charles},
booktitle={2013 10th Working Conference on Mining Software Repositories (MSR)},
pages={207--216},
year={2013},
organization={IEEE}
}</code></pre>

<pre><code>@inproceedings{just2014defects4j,
title={Defects4J: A database of existing faults to enable controlled testing studies for Java programs},
author={Just, Ren{\'e} and Jalali, Darioush and Ernst, Michael D},
booktitle={Proceedings of the 2014 International Symposium on Software Testing and Analysis},
pages={437--440},
year={2014}
}</code></pre>

<pre><code>@article{iyer2018mapping,
title={Mapping language to code in programmatic context},
author={Iyer, Srinivasan and Konstas, Ioannis and Cheung, Alvin and Zettlemoyer, Luke},
journal={arXiv preprint arXiv:1808.09588},
year={2018}
}</code></pre>

<pre><code>@inproceedings{yao2018staqc, title={Staqc: A systematically mined question-code dataset from stack overflow},
author={Yao, Ziyu and Weld, Daniel S and Chen, Wei-Peng and Sun, Huan},
booktitle={Proceedings of the 2018 World Wide Web Conference},
pages={1693--1703},
year={2018}
}</code></pre>

<pre><code>@inproceedings{PanthaplackelETAL20CommentUpdate,
author = {Panthaplackel, Sheena and Nie, Pengyu and Gligoric, Milos and Li, Junyi Jessy and Mooney, Raymond J.},
title = {Learning to Update Natural Language Comments Based on Code Changes},
booktitle = {Association for Computational Linguistics},
pages = {To appear},
year = {2020},
}</code></pre>

<pre><code>@article{feng2020codebert,
title={Codebert: A pre-trained model for programming and natural languages},
author={Feng, Zhangyin and Guo, Daya and Tang, Duyu and Duan, Nan and Feng, Xiaocheng and Gong, Ming and Shou, Linjun and Qin, Bing and Liu, Ting and Jiang, Daxin and others},
journal={arXiv preprint arXiv:2002.08155},
year={2020}
}</code></pre>

# Teams and Conditions
The CodeXGLUE datasets are intended for non-commercial research purposes only to promote advancement in the field of artificial intelligence and related areas, and is made available free of charge without extending any license or other intellectual property rights. The dataset is provided “as is” without warranty and usage of the data has risks since we may not own the underlying rights in the documents. We are not be liable for any damages related to use of the dataset. Feedback is voluntarily given and can be used as we see fit. Upon violation of any of these terms, your rights to use the dataset will end automatically.
If you have questions about use of the dataset or any research outputs in your products or services, we encourage you to undertake your own independent legal review. For other questions, please feel free to contact us.

# LICENSE

given the CodeGLUE project:
1. Contains text-only code
2. Is only for "computational purpose (i.e. machine learning)", and  
3. All the 3rd party dataset used in this project(listed below) was assembled lawfully from publicly accessible sources
It applies to Computational Use of Data Agreement. Pls find all info of Computational Use of Data Agreement from the below table

Computational Use of Data Agreement (C-UDA)
README | Annotated Agreement | Agreement
Computational Use of Data Agreement GitHub repo

