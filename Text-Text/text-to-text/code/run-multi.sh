lr=5e-5
batch_size=32
beam_size=5
source_length=256
target_length=256
data_dir=../data/processed
output_dir=saved_models/multi_model
train_file=$data_dir/train.all.src,$data_dir/train.all.tgt
dev_file=$data_dir/dev.all.src,$data_dir/dev.all.tgt
eval_steps=5000 
train_steps=50000
pretrained_model=xlm-roberta-base


CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
--do_train \
--do_eval \
--using_pretrain_model \
--model_type roberta \
--model_name_or_path $pretrained_model \
--config_name xlm-roberta-base \
--tokenizer_name xlm-roberta-base \
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


beam_size=5
batch_size=32
source_length=256
target_length=256
output_dir=saved_models/multi_model
data_dir=../data/processed
dev_file=$data_dir/dev.all.src,$data_dir/dev.all.tgt
test_file=$data_dir/test.all.src,$data_dir/test.all.tgt
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin


CUDA_VISIBLE_DEVICES=0,1 python3 run.py \
--do_test \
--model_type roberta \
--model_name_or_path xlm-roberta-base \
--config_name xlm-roberta-base \
--tokenizer_name xlm-roberta-base  \
--load_model_path $test_model \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size
 
