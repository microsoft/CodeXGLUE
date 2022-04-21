# CodeXGLUE -- Code2Code Translation

## Task Definition

Code translation aims to migrate legacy software from one programming language in a platform toanother.
In CodeXGLUE, given a piece of Java (C#) code, the task is to translate the code into C# (Java) version. 
Models are evaluated by BLEU scores, accuracy (exactly match), and [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/code-to-code-trans/CodeBLEU.MD) scores.

## Dataset

The dataset is collected from several public repos, including Lucene(http://lucene.apache.org/), POI(http://poi.apache.org/), JGit(https://github.com/eclipse/jgit/) and Antlr(https://github.com/antlr/).

We collect both the Java and C# versions of the codes and find the parallel functions. After removing duplicates and functions with the empty body, we split the whole dataset into training, validation and test sets.

### Data Format

The dataset is in the "data" folder. Each line of the files is a function, and the suffix of the file indicates the programming language.

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ------- | :-------: |
|  Train  |   10,300  |
|  Valid  |      500   |
|   Test  |    1,000  |

## Evaluator

We provide a script to evaluate predictions for this task, and report BLEU scores and accuracy (exactly math score).

### Example

```bash
python evaluator/evaluator.py -ref evaluator/references.txt -pre evaluator/predictions.txt
```

BLEU: 61.08, Acc: 50.0

## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task. 
### Dependency

- python 3.6 or 3.7
- torch>=1.4.0
- transformers>=2.5.0
- pip install scikit-learn

### Fine-tune
Taking Java to C# translation as example:

```shell
cd code
$pretrained_model = the place where you download CodeBERT models e.g. microsoft/codebert-base
$output_dir = the place where you want to save the fine-tuned models and predictions
python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path $pretrained_model \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate lr=5e-5 \
	--train_steps 100000 \
	--eval_steps 5000

```

### Inference

We use full test data for inference. 

```shell
cd code
$output_dir = the place where you want to save the fine-tuned models and predictions
python run.py \
    	--do_test \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path $output_dir/checkpoint-best-bleu/pytorch_model.bin \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--test_filename ../data/test.java-cs.txt.java,../data/test.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--eval_batch_size 16 
```

### Evaluation

(1) Java to C#
```shell
python evaluator/evaluator.py -ref data/test.java-cs.txt.cs -pre code/saved_models/java-cs-model1.output
```
BLEU: 77.46 ; Acc: 56.1

(2) C# to Java
```shell
python evaluator/evaluator.py -ref data/test.java-cs.txt.java -pre code/saved_models/cs-java-model1.output
```
BLEU: 71.99 ; Acc: 57.9

## Result

The results on the test set are shown as below:

Java to C#:

|     Method     |    BLEU    | Acc (100%) |  [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-to-code-trans/CodeBLEU.MD) |  
|    ----------  | :--------: | :-------:  | :-------: |
| Naive copy     |   18.54    |    0.0     |      -    |
| PBSMT      	 |   43.53    |   12.5     |   42.71   |
| Transformer    |   55.84    |   33.0     |   63.74   |
| Roborta (code) |   77.46    |   56.1     |   83.07   |
| CodeBERT   	 | **79.92**  | **59.0**   | **85.10** |

C# to Java:

|     Method     |    BLEU    | Acc (100%) |  [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-to-code-trans/CodeBLEU.MD) | 
|    ----------  | :--------: | :-------:  | :-------: |
| Naive copy     |   18.69    |     0.0    |      -    |
| PBSMT          |   40.06    |    16.1    |   43.48   |
| Transformer    |   50.47    |    37.9    |   61.59   |
| Roborta (code) |   71.99    |    57.9    | **80.18** |
| CodeBERT       | **72.14**  |  **58.0**  |   79.41   |

