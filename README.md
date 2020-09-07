# Introduction

According to [Evans Data Corporation](https://evansdata.com/press/viewRelease.php?pressID=278), there are 23.9 million professional developers in 2019, and the population is expected to reach 28.7 million in 2024. With the growing population of developers, code intelligence, which aims at using AI tools to help enormous software developers improve the productivity of developments, is increasingly important in both communities of software engineering and artificial intelligence. When developers want to find codes written by others with the same intent, [code search](https://arxiv.org/abs/1909.09436) systems could help on automatically retrieving semantically relevant codes given natural language queries. When developers are confused about what tokens to write next, [code completion](https://arxiv.org/abs/1912.00742) systems could help on autocompleting following tokens given contexts of codes. When developers want to implement Java codes with the same function of existing Python codes, [code-to-code translation](https://arxiv.org/abs/2006.03511) systems could help on translating codes from one programming language to another programming language. 

Code intelligence plays a vital role in Microsoft’s mission to empower developers. As highlighted by Satya Nadella at [Build conference 2020](https://mybuild.microsoft.com/sessions/23912de2-1531-4684-b85a-d57ac30af09e), role of developers is more important than ever. GitHub is the home for developers, and Visual Studio Code is the most popular code editor. Microsoft is building the most complete toolchain for developers, bringing together the best of GitHub, Visual Studio and Azure to help developers to go from idea to code and code to cloud. 

Past years have seen the surge of application of statistical models including neural nets in code intelligence tasks. More recently, inspired by the great success of large pre-trained models like [BERT](https://arxiv.org/abs/1810.04805) and [GPT](https://arxiv.org/abs/1908.09203) in natural language processing, pre-trained models learnt from big programming language data including [IntelliCode](https://arxiv.org/pdf/2005.08025.pdf) and [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) obtain further surprising improvements on code understanding and generation problems. However, the area of code intelligence lacks a benchmark that covers a wide range of tasks. We have seen that a diversified benchmark dataset is significant for the growth of an area, like [ImageNet](http://image-net.org/) for CV and [GLUE](https://gluebenchmark.com/) for NLP. 

To address this, we introduce CodeXGLUE, a benchmark dataset and open challenge for code intelligence. It includes a collection of code intelligence tasks and a platform for model evaluation and comparison. CodeXGLUE stands for General Language Understanding Evaluation benchmark for CODE. It includes 14 datasets for 10 diversified code intelligence tasks covering code-code (clone detection, defect detection, cloze test, code completion, code refinement, and code-to-code translation), text-code (natural language code search, text-to-code generation), code-text (code summarization) and text-text (documentation translation) scenarios. With CodeXGLUE, we seek to support the development of models that can be applied to various code intelligence problems, and finally increase the development productivity for software developers.  

A brief summary of CodeXGLUE is given below, including tasks, datasets, baseline systems, etc. Datasets highlighed in BLUE are newly introduced. 
![A brief summary of CodeXGLUE, including tasks, datasets, baseline systems, etc.](https://github.com/microsoft/CodeXGLUE/blob/main/tasks.jpg)

We provide three baseline models to support these tasks, including BERT-style pre-trained model (i.e. [CodeBERT](https://github.com/microsoft/CodeBERT)) which is good at understanding problems, GPT-style pre-trained model which we call CodeGPT to support completion and generation problems, and Encoder-Decoder framework that supports sequence-to-sequence generation problems. 
Three pipelines including CodeBERT, CodeGPT and Encoder-Decoder are given below.
![baselines](https://github.com/microsoft/CodeXGLUE/blob/main/baselines.jpg)

# Relevant Links
[Leaderboard](https://microsoft.github.io/CodeXGLUE/) | [CodeXGLUE paper](arxivpaper-to-be-added)

# Tasks and Datasets

1.	**Clone detection**. A model is tasked with measure the semantic similarity between codes. Two existing datasets are included. One is for binary classification between codes and the another is for retrieving semantically similar code given code as the query. 
2.	**Defect detection**. A model is tasked with identifying whether a source code is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack. An existing dataset is included.
3.	**Cloze test**. A model is tasked with predicting the masked token from a code, formulated as a multi-choice classification problem. Two datasets are newly created, one with candidates from the (filtered) vocabulary and the other with candidates among “max” and “min”. 
4.	**Code completion**. A model is tasked with predicting following tokens given contexts of codes. Both token-level and line-level completion are covered. Token-level task is analogous to language modeling, and we include two influential datasets here. Line-level datasets are newly created to test model’s ability to autocomplete a line. 
5.	**Code translation**.  A model is tasked to translate the code in one programming language to the code in another one. A dataset between Java and C# is newly created.
6.	**Code search**. A model is tasked to measure the semantic similarity between text and code. In retrieval scenario, a test set is newly created where function names and variables in test sets are replaced to test the generalization ability of a model. In text-code classification scenario, a test set where natural language queries come from Bing query log is created to test on real user queries.
7.	**Code refinement**. A model is tasked to try to automatically refine the code, which could be buggy or complex. An existing dataset is included.
8.	**Text-to-code generation**. A model is tasked to generate a code given natural language description. An existing dataset is included.
9.	**Code summarization**. A model is tasked to generate natural language comments for a code. Existing datasets are included.  
10.	**Documentation translation**. A model is tasked to translate code documentation between human languages. A dataset, focusing on low-resource multilingual translation, is newly created.

# LICENSE
Our codes follow MIT License.

Our datasets follow Computational Use of Data Agreement (C-UDA) License.

# Reference
If you use this code or CodeXGLUE, please consider citing us.
<pre><code>@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}</code></pre>

This research was conducted by Ambrosio Blanco, Colin Clement, Shao Kun Deng, Dawn Drain, Nan Duan, Shengyu Fu, Ming Gong, Daya Guo, Niklas Gustafsson, Junjie Huang, Daxin Jiang,  Shujie Liu, Shuai Lu, Shuo Ren, Linjun Shou, Neel Sundaresan, Alexey Svyatkovskiy, Duyu Tang, Michele Tufano, Lidong Zhou, Long Zhou, Ming Zhou (authors in alphabet order). 

Feel free to contact Duyu Tang (dutang@microsoft.com) and Shujie Liu (shujliu@microsoft.com) with any questions or comments.
