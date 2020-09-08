
# CodeBLEU -- a Weighted Syntactic and Semantic BLEU for Code Synthesis Evaluation

## Introduction

The code synthesis receives more and more attention recently, especially with the powerful machine learning techniques. 
In order to facilitate the code synthesis research, we propose a new evaluation metric CodeBLEU, which can not only consider the surface match similar with the original
BLEU, but can also consider the grammatical correctness and the logic correctness, leveraging the abstract syntax tree and the data-flow structure.


## Motivation

### The current metrics (BLEU, acc) for code synthesis tasks, such as code translation, are not suitable enough.

       (1) BLEU: just consider n-gram match, ignore code structure

       (2) Accuracy (exactly match): too strict for code evaluation

### Program language vs. Natural language

       (1) Limited keywords vs. millions of words.

       (2) Tree structure vs. sequential structure.

       (3) Unique instructions vs. ambiguous semantic.

## Methods

An ideal evaluation metric should consider the grammatical correctness and the logic correctness.
![We propose weighted n-gram match and syntactic AST match to measure grammatical correctness, and introduce semantic data-flow match to calculate logic correctness](https://github.com/microsoft/CodeXGLUE/blob/main/code-to-code-trans/CodeBLEU.jpg)


## Experiments

We conduct experiments on three code generation tasks (text-to-code, code translation, and code refinement) to evaluate the effectiveness of CodeBLEU.

### Steps

（1）Translate or generate codes with different systems

（2）Calculate the CodeBLEU of the code synthesis results

（3）Carry out human evaluation on code synthesis results.

（4）Calculate the correlation coefficient of the CodeBLEU and human evaluation scores

### 

### Results


| Metric     | Text-to-code | Code Translation | Code Refinement |   
| ----------- |   :-------: |  :--------: |  :--------: | 
| BLEU       |    0.967    |    0.940    |  0.923 |
| ACC        |    0.912    |   0.968     |   0.999 |
| CodeBLEU   |  0.977 (+1%) | 0.970 (+3%) | 0.979 (+5.6%) |


### Abation Study


| Components     | Text-to-code | Code Translation | Code Refinement |   
| ----------- |   :-------: |  :--------: |  :--------: | 
| n-gram       |   0.967    |    0.940    |  0.923   |
| weighted n-gram |    0.960    |   0.934  |   0.985 |
| syntactic AST   |  0.985      | 0.977    | 0.967 |
| semantic data-flow |  0.978   | 0.974    | 0.983 |
| CodeBLEU           |  0.977   | 0.970    | 0.979 |



# Conclusion



