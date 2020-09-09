lr=5e-5
batch_size=32
beam_size=5
source_length=514
target_length=514
data_dir=data/java2cs
output_dir=saved_models/java-cs_model1
train_file=$data_dir/train.java-cs.txt.java $data_dir/train.java-cs.txt.cs
dev_file=$data_dir/valid.java-cs.txt.java $data_dir/valid.java-cs.txt.cs
eval_steps=5000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=100000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=CodeBERT1 #CodeBERT: path to CodeBERT. Roberta: roberta-base

CUDA_VISIBLE_DEVICES=0,1 python run.py \
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


batch_size=16
source_length=514
target_length=514
output_dir=saved_models/java-cs_model1
data_dir=data/java2cs
dev_file=$data_dir/valid.java-cs.txt.java $data_dir/valid.java-cs.txt.cs
test_file=$data_dir/test.java-cs.txt.java $data_dir/test.java-cs.txt.cs
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

CUDA_VISIBLE_DEVICES=0,1 python run.py \
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
