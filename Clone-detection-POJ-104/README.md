# CodeXGLUE -- Clone Detection (POJ-104)

Here is the pipeline for clone detection task on the POJ-104 dataset.


## Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0
- pip install gdown


## Data Preprocess

1.Download dataset from [website](https://drive.google.com/file/d/0B2i-vWnOu7MxVlJwQXN6eVNONUU/view?usp=sharing) or run the following command:

```shell
cd dataset
gdown https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU
tar -xvf programs.tar.gz
cd ..
```

2.Preprocess data

```shell
cd dataset
python preprocess.py
cd ..
```

## Fine-tune

To fine-tune CodeBERT on the dataset

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 
```


## Evaluation

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
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 
```

## Result

The results on the test set are shown as below:

| Method           | MAP@R (%) |
| ---------------- | :-------: |
| code2vec         |   1.98    |
| NCC              |   39.95   |
| NCC-w/0-inst2vec |   54.19   |
| Aroma-Dot        |   52.08   |
| Aroma-Cos        |   55.12   |
| MISIM-GNN        |   82.45   |
| CodeBERT         | **84.13** |

