sudo ~/anaconda3/bin/pip install tree-sitter
pretrained_model=CodeBERT
CUDA_VISIBLE_DEVICES=2,3 ~/anaconda3/bin/python run.py \
    --do_train \
    --do_eval \
    --model_type roberta \
    --model_name_or_path $pretrained_model \
    --config_name roberta-base \
    --tokenizer_name roberta-base \
    --train_source_filename ../data/small/train.buggy-fixed.buggy \
    --train_target_filename ../data/small/train.buggy-fixed.buggy  \
    --dev_source_filename ../data/small/valid.buggy-fixed.buggy \
    --dev_target_filename ../data/small/valid.buggy-fixed.buggy  \
    --output_dir model3_bug_fixed_small \
    --max_source_length 256 \
    --max_target_length 256 \
    --max_dfg_length 64 \
    --beam_size 5 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5  \
    --train_steps 100000  \
    --eval_steps 5000 \
    --language java

CUDA_VISIBLE_DEVICES=2,3 ~/anaconda3/bin/python run.py \
    --do_test \
    --load_model_path model3_bug_fixed_small/checkpoint-best-bleu/pytorch_model.bin \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --config_name roberta-base \
    --tokenizer_name roberta-base \
    --test_source_filename ../data/small/test.buggy-fixed.buggy \
    --test_target_filename ../data/small/test.buggy-fixed.buggy  \
    --output_dir model3_bug_fixed_small \
    --max_source_length 256 \
    --max_target_length 256 \
    --max_dfg_length 64 \
    --beam_size 5 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --train_steps 100000  \
    --eval_steps 5000 \
    --language java
    