# CodeXGLUE -- Clone Detection (BCB)

Here is the pipeline for clone detection task on the BigCloneBench dataset.


## Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0


## Fine-tune

To fine-tune encoder-decoder on the dataset

```shell
cd code
mkdir data
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
    --epoch 1 \
    --block_size 512 \
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
    --train_data_file=../dataset/train.txt \
    --eval_data_file=../dataset/valid.txt \
    --test_data_file=../dataset/test.txt \
    --epoch 1 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 
```

## Result

The results on the test set are shown as below:

| Method     |    F1     |
| ---------- | :-------: |
| Deckard    |   0.03    |
| RtvNN      |   0.01    |
| CDLH       |   0.82    |
| ASTNN      |   0.93    |
| FA-AST-GMN |   0.95    |
| CodeBERT   | **0.965** |
