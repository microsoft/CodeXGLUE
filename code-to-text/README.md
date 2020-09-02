# CodeXGLUE -- Code-To-Text

Here is the pipeline for code-to-text generation task.


## Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0


## Data Preprocess

1.Download and preprocess dataset:

```shell
unzip dataset
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

## Fine-tune

To fine-tune encoder-decoder on the dataset

```shell
lang=ruby #programming language
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
eval_steps=400 #400 for ruby, 600 for javascript, 1000 for others
train_steps=20000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps 
```


## Evaluation

```shell
batch_size=128
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
```

## Result

The results on the test set are shown as below:

| Model       |   Ruby    | Javascript |    Go     |  Python   |   Java    |    PHP    |  Overall  |
| ----------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Seq2Seq     |   9.64    |   10.21    |   13.98   |   15.93   |   15.09   |   21.08   |   14.32   |
| Transformer |   11.18   |   11.59    |   16.38   |   15.81   |   16.26   |   22.12   |   15.56   |
| RoBERTa     |   11.17   |   11.90    |   17.72   |   18.14   |   16.47   |   24.02   |   16.57   |
| CodeBERT    | **12.16** | **14.90**  | **18.07** | **19.06** | **17.65** | **25.16** | **17.83** |
