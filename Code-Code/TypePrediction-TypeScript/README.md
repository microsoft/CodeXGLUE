# CodeXGLUE -- Type Prediction -- TypeScript

## Task Definition

Given a sequence of source code, the task is to predict the correct type for a particular variable, parameter, or
function. Type prediction is an important task for software developement, especially in dynamically-typed environments,
which can benefit from stronger type checking while maintaining the advantages of dynamic typing. We treat the task as
sequence tagging task, similar to Named-Entity Recognition (NER) in NLP.

### Dataset

The dataset originates from the MSR '22
paper [ManyTypes4TypeScript: A Comprehensive TypeScript Dataset for Sequence-Based Type Inference](). The dataset is
available for download on Zenodo.

### Download and Preprocess

1. Download dataset either using wget per below or by clicking on
   the [link](https://zenodo.org/record/6387001/files/ManyTypes4TypeScript.tar.gz?download=1). Unzip and run the
   preprocess script with chosen parameters like type vocabulary size. To use the default type vocabulary size of 50k
   types, train#.jsonl, test.jsonl, valid.jsonl are already available in the main directory.

```shell
cd CodeXGLUE/Code-Code/TypePrediction-TypeScript/dataset
wget https://zenodo.org/record/6387001/files/ManyTypes4TypeScript.tar.gz?download=1 -O ManyTypes4TypeScript.tar.gz
tar -xvzf ManyTypes4TypeScript.tar.gz

```

2. Install required packages. Use

```shell
pip install -r requirements.txt
```

3. Preprocess dataset Use the process_datafiles.py in dataset folder.

```shell
python process_datafiles.py -v <vocab-size> 
```

A vocabulary of 50,000 was used in ManyTypes4TypeScript paper.

### Data Format

After preprocessing dataset there should be five jsonl files. train0.jsonl, train1.jsonl, train2.jsonl, test.jsonl,
valid.jsonl. Train is split to accomodate Git LFS.

Each line represents a file. The data fields are the same among all splits.

|field name. | type        |               description                  |
|------------|-------------|--------------------------------------------|
|tokens      |list[string] | Sequence of tokens (word tokenization)     |
|labels      |list[string] | A list of corresponding types              |
|url         |string       | Repository URL                             |
|path        |string       | Original file path that contains this code |
|commit_hash |string       | Commit identifier in the original project  |
|file        |string       | File name                                  |

### Data Splits

|   name   |  train   |test|   validation  |
|---------:|---------:|---------:|--------:|
|projects  |  11,413 (81.8%) | 1,336 (9.58%)  | 1,204 (8.62%) |
|files     |  486,477 (90.16%) |28,045 (5.20%)|  25,049 (4.64%)|
|sequences |  1,727,927 (91.95%)| 81,627 (4.34%)| 69,652 (3.71%)  |
|types     |  8,696,679 (95.33%) |  224,415  (2.46%)|  201,428 (2.21%) |

## Evaluator

We provide a script to evaluate predictions for this task, and report accuracy score. It can be found at
evaluator/evaluator.py. The train.py in the code directory also has a built in evaluator and reports scores before
generating a list of predictions.

### Example

```shell
python evaluator/evaluator.py -a evaluator/gold_labels.txt -p evaluator/example-codebert-predictions.txt
```

{'Acc': 0.6280417975625515}

### Input predictions

A prediction file that has predictions in the form of

```shell
index	type
```

See example-codebert-predictions.txt.

## Pipeline-CodeBERT

We also provide a pipeline that finetunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task; see train.py.
For models with torch weights, the model state_dict can be loaded traditionally with torch.load()

### Train (Finetune)

The train.py file will train and eval by default. The validation dataset can be replaced in the evalulation. We provide
a pipeline for finetuning huggingface models like [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf); this can be expanded
to any huggingface pretrained encoding model.

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py --output_dir type-model --train_batch_size=36 --eval_batch_size=16
```

### Inference

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py --output_dir type-model --do_train=False --eval_batch_size=16
```

### Evaluation

```shell
python evaluator/evaluator.py -a evaluator/gold_labels.txt -p evaluator/example-codebert-predictions.txt
```

{'Acc': 0.6280417975625515}

## Results

The results from the paper are listed below. Top 100 indicates the performance across the top 100 most frequently
occuring types.

| Model | Top 100 | | | | Overall | | | |
| --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
| |Precision | Recall | F1 | Accuracy | Precision | Recall | F1 | Accuracy|
| [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) | 84.58 | 85.98 | 85.27 | 87.94 | 59.34 | 59.80 | 59.57 | 61.72|
| [GraphCodeBERT](https://arxiv.org/pdf/2009.08366.pdf)| **84.67**  | **86.41**  | **85.53** | **88.08** | **60.06** | **61.08** | **60.57** | **62.51** |
| [CodeBERTa](https://huggingface.co/huggingface/CodeBERTa-small-v1) | 81.31 | 82.72| 82.01  | 85.94|56.57|56.85|56.71| 59.81|
| [PolyGot](https://arxiv.org/pdf/2112.02043.pdf) | 84.45 | 85.45 | 84.95 | 87.72 | 58.81 | 58.91 | 58.86 | 61.29   |
| [GraphPolyGot](https://arxiv.org/pdf/2112.02043.pdf)  | 83.80  | 85.23 |  84.51  | 87.40 | 58.36 | 58.91 | 58.63 | 61.00  |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) | 82.03 | 83.81 | 82.91  | 86.25 | 57.45 | 57.62 | 57.54 | 59.84|
| [BERT](https://arxiv.org/pdf/1810.04805.pdf) | 80.04  | 81.50 | 80.76 | 84.97 | 54.18 | 54.02 | 54.10 | 57.52 |

## Reference

Please cite the original paper.
<pre><code>
</code></pre>


