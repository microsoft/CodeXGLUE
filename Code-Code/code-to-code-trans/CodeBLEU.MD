
# CodeBLEU -- a Weighted Syntactic and Semantic BLEU for Code Synthesis Evaluation

## Introduction

The code synthesis receives more and more attention recently, especially with the powerful machine learning techniques. 
In order to facilitate the code synthesis research, we propose a new evaluation metric CodeBLEU, which can not only consider the surface match similar with the original
BLEU, but can also consider the grammatical correctness and the logic correctness, leveraging the abstract syntax tree and the data-flow structure.


## Motivation

(1) The current metrics (BLEU, acc) for code synthesis tasks, such as code translation, are not suitable enough.

       - BLEU: just consider n-gram match, ignore code structure.

       - Accuracy (exactly match): too strict for code evaluation.

(2) Program language vs. Natural language

       - Limited keywords vs. millions of words.

       - Tree structure vs. sequential structure.

       - Unique instructions vs. ambiguous semantic.

## Methods

An ideal evaluation metric should consider the grammatical correctness and the logic correctness.
We propose weighted n-gram match and syntactic AST match to measure grammatical correctness, and introduce semantic data-flow match to calculate logic correctness.
![CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/code-to-code-trans/CodeBLEU.jpg)

### Examples

Here we will give two toy examples and show the qualitative advantages of CodeBLEU compared with the traditional BLEU score.

![Example](https://github.com/microsoft/CodeXGLUE/blob/main/code-to-code-trans/example.png)




## Experiments

We conduct experiments on three code generation tasks (text-to-code, code translation, and code refinement) to evaluate the effectiveness of CodeBLEU.

### Steps

（1）Translate or generate codes with different systems.

（2）Calculate the CodeBLEU of the code synthesis results.

（3）Carry out human evaluation on code synthesis results.

（4）Calculate the correlation coefficient of the CodeBLEU and human evaluation scores.

### 

### Results 

(1) The results of all baselines of the given three tasks evaluated by BLEU, accuracy (exactly match), CodeBLEU and human evaluation scores.

-Text-to-code
| System     | BLEU        | ACC        | CodeBLEU | Human score |   
| ----------- | :-------: | :--------: | :--------: | :--------:| 
| Sys1       |    12.02    |    3.05     |  18.04    |  1.88    |
| Sys2        |    16.82    |   10.50     |  21.71   |  1.99    |
| Sys3       |    21.18    |    17.35    |   24.95   |  2.56    |
| Sys4       |    26.45    |    20.10    |    30.96  |  3.13    |

-Code Translation
| System     | BLEU        | ACC        | CodeBLEU | Human score |   
| ----------- | :-------: | :--------: | :--------: | :--------:| 
| Sys1       |    44.53    |    13.2     |  45.71    |  3.25    |
| Sys2        |    54.84    |   31.75     |  61.14   |  3.77    |
| Sys3       |    80.18    |    60.2    |   82.74   |  4.04    |
| Sys4       |    81.44    |    63.5    |    84.75  |  4.25    |

-Code Refinement
| System     | BLEU        | ACC        | CodeBLEU | Human score |   
| ----------- | :-------: | :--------: | :--------: | :--------:| 
| Sys1       |    90.35    |    3.00     |  80.81    |  1.38    |
| Sys2        |   91.40    |   7.01     |  82.16   |  1.54    |
| Sys3       |    92.80    |    17.6    |   83.85   |  2.02    |


(2) Comparison of the Pearson correlation coefficients between human evaluation scores and three different metrics.

| Metric     | Text-to-code | Code Translation | Code Refinement |   
| ----------- |   :-------: |  :--------: |  :--------: | 
| BLEU       |    0.967    |    0.940    |  0.923 |
| ACC        |    0.912    |   0.968     |   **0.999** |
| CodeBLEU   |  **0.977** (+1.0%) | **0.970** (+3.0%) | 0.979 (+5.6%) |


### Abation Study

Resutls of the respective Pearson correlation between the human evaluation scores and the scores given by different components.

| Components     | Text-to-code | Code Translation | Code Refinement |   
| ----------- |   :-------: |  :--------: |  :--------: | 
| n-gram       |   0.967    |    0.940    |  0.923   |
| weighted n-gram |    0.960    |   0.934  |   **0.985** |
| syntactic AST   |  **0.985**      | **0.977**    | 0.967 |
| semantic data-flow |  **0.978**   | **0.974**    | **0.983** |
| CodeBLEU           |  0.977   | 0.970    | 0.979 |





