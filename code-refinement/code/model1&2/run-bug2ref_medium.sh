
lr=5e-5
batch_size=8
beam_size=5
source_length=256
target_length=256
data_dir=../data/medium
output_dir=saved_models/bug-fixed_model1_medium
train_file=$data_dir/train.buggy-fixed.buggy,$data_dir/train.buggy-fixed.fixed
dev_file=$data_dir/valid.buggy-fixed.buggy,$data_dir/valid.buggy-fixed.fixed
eval_steps=5000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=100000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=CodeBERT1 #CodeBERT: path to CodeBERT. Roberta: roberta-base

CUDA_VISIBLE_DEVICES=2,3 ~/anaconda3/bin/python run.py \
--do_train \
--do_eval \
--model_type roberta \
--model_name_or_path $pretrained_model \
--config_name roberta-base \
--tokenizer_name roberta-base \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--train_steps $train_steps \
--eval_steps $eval_steps


batch_size=8
source_length=256
target_length=256
output_dir=saved_models/bug-fixed_model1_medium
data_dir=../data/medium
dev_file=$data_dir/valid.buggy-fixed.buggy,$data_dir/valid.buggy-fixed.fixed
test_file=$data_dir/test.buggy-fixed.buggy,$data_dir/test.buggy-fixed.fixed
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

CUDA_VISIBLE_DEVICES=2,3 ~/anaconda3/bin/python run.py \
--do_test \
--model_type roberta \
--model_name_or_path roberta-base \
--config_name roberta-base \
--tokenizer_name roberta-base  \
--load_model_path $test_model \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size 
