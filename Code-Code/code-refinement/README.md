# CodeXGLUE -- Code Refinement

## Task Definition

Code refinement aims to automatically fix bugs in the code, which can contribute to reducing the cost of bug-fixes for developers.
In CodeXGLUE, given a piece of Java code with bugs, the task is to remove the bugs to output the refined code. 
Models are evaluated by BLEU scores, accuracy (exactly match) and [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/code-to-code-trans/CodeBLEU.MD).

## Dataset

We use the dataset released by this paper(https://arxiv.org/pdf/1812.08693.pdf). The source side is a Java function with bugs and the target side is the refined one. 
All the function and variable names are normalized. Their dataset contains two subsets ( i.e.small and medium) based on the function length.

### Data Format

The dataset is in the "data" folder. Each line of the files is a function.

### Data Statistics

Data statistics of this dataset are shown in the below table:

|         | #Examples | #Examples |
| ------- | :-------: | :-------: |
|         |   Small   |   Medium  |
|  Train  |   46,680  |   52,364  |
|  Valid  |    5,835  |    6,545  |
|   Test  |    5,835  |    6,545  |

## Evaluator

We provide a script to evaluate predictions for this task, and report BLEU scores and accuracy (exactly math score).

### Example

```bash
python evaluator/evaluator.py -ref evaluator/references.txt -pre evaluator/predictions.txt
```

BLEU: 79.03, Acc: 40.0

## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task. 
### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0
- pip install scikit-learn

### Fine-tune
Taking the "small" subset as example:

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
	--train_filename ../data/train.buggy-fixed.buggy,../data/train.buggy-fixed.fixed \
	--dev_filename ../data/valid.buggy-fixed.buggy,../data/valid.buggy-fixed.fixed \
	--output_dir $output_dir \
	--max_source_length 256 \
	--max_target_length 256 \
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
	--dev_filename ../data/valid.buggy-fixed.buggy,../data/valid.buggy-fixed.fixed \
	--test_filename ../data/test.buggy-fixed.buggy,../data/test.buggy-fixed.fixed \
	--output_dir $output_dir \
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--eval_batch_size 16 
```

### Evaluation

Small:
```shell
python evaluator/evaluator.py -ref data/small/test.buggy-fixed.fixed -pre code/saved_models/small-model.output
```
BLEU: 77.42 ; Acc: 16.4

Medium: 
```shell
python evaluator/evaluator.py -ref data/medium/test.buggy-fixed.fixed -pre code/saved_models/medium-model.output
```
BLEU: 91.07 ; Acc: 5.16

## Result

The results on the test set are shown as below:

Small:

| Method     |    BLEU   | Acc (100%) |  [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-to-code-trans/CodeBLEU.MD)  |  
| ---------- | :-------: | :-------:  | :-------:  |
| Naive copy |   78.06   |     0.0    |     -      |
| LSTM       |   76.76   |    10.0    |     -      |
| Transformer|   77.21   |    14.7    |    73.31   | 
| CodeBERT   | **77.42** |  **16.4**  |  **75.58** |

Medium:

| Method     |    BLEU   | Acc (100%) |  [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-to-code-trans/CodeBLEU.MD)  |  
| ---------- | :-------: | :-------:  | :-------:  |
| Naive copy |   90.91   |    0.0     |     -      |
| LSTM       |   72.08   |    2.5     |     -      |
| Transformer|   89.25   |    3.7     |   81.72    |
| CodeBERT   | **91.07** |  **5.16**  | **87.52**  |  

# Reference
<pre><code>@article{tufano2019empirical,
  title={An empirical study on learning bug-fixing patches in the wild via neural machine translation},
  author={Tufano, Michele and Watson, Cody and Bavota, Gabriele and Penta, Massimiliano Di and White, Martin and Poshyvanyk, Denys},
  journal={ACM Transactions on Software Engineering and Methodology (TOSEM)},
  volume={28},
  number={4},
  pages={1--29},
  year={2019},
  publisher={ACM New York, NY, USA}
}</code></pre>

