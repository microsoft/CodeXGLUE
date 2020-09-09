sudo ~/anaconda3/bin/pip install tree-sitter
pretrained_model=CodeBERT
CUDA_VISIBLE_DEVICES=0,1 ~/anaconda3/bin/python run.py \
    --do_train \
    --do_eval \
    --model_type roberta \
    --model_name_or_path $pretrained_model \
    --config_name roberta-base \
    --tokenizer_name roberta-base \
    --train_source_filename ../data/train.java-cs.txt.java \
    --train_target_filename ../data/train.java-cs.txt.cs  \
    --dev_source_filename ../data/valid.java-cs.txt.java \
    --dev_target_filename ../data/valid.java-cs.txt.cs  \
    --output_dir save_models/model3_java_cs \
    --max_source_length 256 \
    --max_target_length 256 \
    --max_dfg_length 64 \
    --beam_size 1 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --train_steps 150000  \
    --eval_steps 5000 \
    --language java

CUDA_VISIBLE_DEVICES=0,1 ~/anaconda3/bin/python run.py \
    --do_test \
    --load_model_path save_models/model3_java_cs/checkpoint-best-bleu/pytorch_model.bin \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --config_name roberta-base \
    --tokenizer_name roberta-base \
    --test_source_filename ../data/test.java-cs.txt.java \
    --test_target_filename ../data/test.java-cs.txt.cs  \
    --output_dir model3_java_cs \
    --max_source_length 256 \
    --max_target_length 256 \
    --max_dfg_length 64 \
    --beam_size 5 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --train_steps 150000  \
    --eval_steps 5000 \
    --language java