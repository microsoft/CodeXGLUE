# This repo is under construction...

# Introduction
This repo includes datasets and codes for CodeXGLUE, a collection of code intelligence tasks and a platform for model evaluation and comparison. CodeXGLUE stands for General Language Understanding Evaluation benchmark for CODE. It includes 14 datasets for 10 diversified programming language tasks covering code-code (clone detection, defect detection, cloze test, code completion, code refinement, and code-to-code translation), text-code (natural language code search, text-to-code generation), code-text (code summarization) and text-text (documentation translation) scenarios. We provide three baseline models to support these tasks, including BERT-style pre-trained model (i.e. [CodeBERT](https://github.com/microsoft/CodeBERT)) which is good at understanding problems, GPT-style pre-trained model which we call CodeGPT to support completion and generation problems, and Encoder-Decoder framework that supports sequence-to-sequence generation problems.

A brief summary of CodeXGLUE is given below, including tasks, datasets, baseline systems, etc. Datasets highlighed in BLUE are contributed by Microsoft.
![A brief summary of CodeXGLUE, including tasks, datasets, baseline systems, etc.](https://github.com/microsoft/CodeXGLUE/blob/main/tasks.jpg)

# Relevant Links
[Leaderboard](https://microsoft.github.io/CodeXGLUE/) | [CodeXGLUE paper](arxivpaper-to-be-added)

# Tasks and Datasets

## Clone Detection 
We have two subtasks to measure the semantic equivalence between codes. 

Dataset for the first subtask is Clone-detection-BigCloneBench. Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by classification accuracy. The dataset comes from <code>\cite{xxx}</code>, and we follow the training/dev/testing split of <code>\cite{}</code>.

Dataset for the second subtask is Clone-detection-POJ-104. Given a code as the input, the task is to retrieve semantically simliar codes from a collection of codes.  The dataset comes from <code>\cite{}</code>, and we follow the split of <code>\cite{}</code>. Models are evaluated by xxx. 


## Defect Detection 
The task is to detect whether a source code is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack.
We follow the original training/dev/testing split of Defects4J <code>\cite{} </code> dataset. 

## Cloze Test 

## Code Completion 
We have both token-level and line-level completion tasks. ...

## Code Refinement 

## Code Translation

## Natural Language Code Search 
We have two subtasks to measure semantic similarity between text and code. 

Dataset for the first subtask is NL-code-search-Adv. Given a text as the input, the task is to find the most semantically relevant code in Python from a collection of codes. The training set come from CodeSearchNet <code>\cite{}</code>. Texts are function docstrings which serve as natural language queries. Although deep neural network models like CodeBERT perform well on the original test data with the same data distribution, we observe that the performance drops dramatically once function names and variables are artificially replaced. To test the generalization ability of models, we create dev and test sets, in which data originally comes from CodeSearchNet but the function names and variables are replaced with default values.

Dataset for the second subtask is NL-code-search-WebQuery. Given a text-code pair as the input, the task is to do binary classification (0/1), where 1 stands for relevant and 0 for irrelevant. The training and dev sets come from StacQC <code>\cite</code>, in which natural language queries are questions from StackOverflow. 

## Text-to-Code Generation 

## Code Summarization 

## Documentation Translation 


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

# LICENSE
Our codes follow MIT License.

Our datasets follow Computational Use of Data Agreement (C-UDA) License.
