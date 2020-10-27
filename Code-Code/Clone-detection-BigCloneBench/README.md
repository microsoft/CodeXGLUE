# CodeXGLUE -- Clone Detection (BCB)

## Task Definition

Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.

## Dataset

The dataset we use is [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) and filtered following the paper [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf).

### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the function

   - **idx:** index of the example

2. train.txt/valid.txt/test.txt provide examples, stored in the following format:    idx1	idx2	label

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  901,028  |
| Dev   |  415,416  |
| Test  |  415,416  |

## Evaluator

We provide a script to evaluate predictions for this task, and report F1 score

### Example

```bash
python evaluator/evaluator.py -a evaluator/answers.txt -p evaluator/predictions.txt
```

{'Recall': 0.25, 'Prediction': 0.5, 'F1': 0.3333333333333333}

### Input predictions

A predications file that has predictions in TXT format, such as evaluator/predictions.txt. For example:

```b
13653451	21955002	0
1188160	8831513	1
1141235	14322332	0
16765164	17526811	1
```

## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task. 
### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0
- pip install scikit-learn

### Fine-tune

We only use 10% training data to fine-tune and 10% valid data to evaluate.


```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
```

### Inference

We use full test data for inference. 

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee test.log
```

### Evaluation

```shell
python ../evaluator/evaluator.py -a ../dataset/test.txt -p saved_models/predictions.txt
```

{'Recall': 0.9687694680849823, 'Prediction': 0.9603497142447242, 'F1': 0.9645034096215225}

## Result

The results on the test set are shown as below:

| Method     | Precision |  Recall   |    F1     |
| ---------- | :-------: | :-------: | :-------: |
| [Deckard](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?referer=https://scholar.google.com/&httpsredir=1&article=2010&context=sis_research)    |   0.93    |   0.02    |   0.03    |
| [RtvNN](https://tufanomichele.com/publications/C5.pdf)      |   0.95    |   0.01    |   0.01    |
| [CDLH](https://www.ijcai.org/Proceedings/2017/0423.pdf)       |   0.92    |   0.74    |   0.82    |
| [ASTNN](https://ieeexplore.ieee.org/abstract/document/8812062)      |   0.92    |   0.94    |   0.93    |
| [FA-AST-GMN](https://arxiv.org/pdf/2002.08653.pdf) |   0.96    |   0.94    |   0.95    |
| [TBBCD](http://taoxie.cs.illinois.edu/publications/icpc19-clone.pdf)      |   0.94    |   0.96    |   0.95    |
| [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf)   | **0.960** | **0.969** | **0.965** |


## Reference
<pre><code>@inproceedings{svajlenko2014towards,
  title={Towards a big data curated benchmark of inter-project code clones},
  author={Svajlenko, Jeffrey and Islam, Judith F and Keivanloo, Iman and Roy, Chanchal K and Mia, Mohammad Mamun},
  booktitle={2014 IEEE International Conference on Software Maintenance and Evolution},
  pages={476--480},
  year={2014},
  organization={IEEE}
}

@inproceedings{wang2020detecting,
  title={Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree},
  author={Wang, Wenhan and Li, Ge and Ma, Bo and Xia, Xin and Jin, Zhi},
  booktitle={2020 IEEE 27th International Conference on Software Analysis, Evolution and Reengineering (SANER)},
  pages={261--271},
  year={2020},
  organization={IEEE}
}</code></pre>
