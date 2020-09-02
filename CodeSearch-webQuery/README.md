# CodeXGLUE -- CodeSearch-webQuery

Here is the pipeline for NL-code-search-WebQuery task.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.1.0


## Data

The  training and validation sets of CodeSearch-webQuery data are collected from StaQC, where each instance contains a StackOverflow title, a code snippet and a 0/1 annotation of whether the code answers the title.

As for test set, we include the original test set of StaQC. Also to apply to a wider scenario, we create a new test set where each instance contains a Bing Search query, a python code function and a 0/1 annotation of whether the code answers the query. The newly created test set is in `./data/test_search.txt` .

Download StaQC dataset and preprocess data in this folder:

```shell
git clone https://gisthub.com/LittleYUYU/StackOverflow-Question-Code-Dataset.git
python data_preprocess.py
```

The processed data can be found in `./data` directory. 

Data statistics of CodeSearch-webQuery are shown in the below table:

|             | #Examples |
| ----------- | :-------: |
| train       |   2,932   |
| validation  |    976    |
| test_staqc  |    976    |
| test_search |   1,046   |


## Fine-tuning

You can fine-tune a pre-trained model by the following command:

```shell
python run_classifier.py \
			--model_type roberta \
			--task_name staqc \
			--do_train \
			--do_eval \
			--eval_all_checkpoints \
			--train_file train.txt \
			--dev_file dev.txt \
			--max_seq_length 200 \
			--per_gpu_train_batch_size 16 \
			--per_gpu_eval_batch_size 16 \
			--learning_rate 1e-5 \
			--num_train_epochs 10 \
			--gradient_accumulation_steps 1  \
			--warmup_steps 100 \
			--overwrite_output_dir \
			--data_dir ./data/ \
			--output_dir ./model_testfinal/ \
			--model_name_or_path microsoft/codebert-base \
			--config_name roberta-base
```

## Evaluation

To test on the StaQC test set, you run the following command:

```shell
python run_classifier.py \
			--model_type roberta \
			--task_name staqc \
			--do_predict \
			--test_file test_staqc.txt \
			--max_seq_length 200 \
			--per_gpu_eval_batch_size 2 \
			--data_dir ./data/ \
			--output_dir ./model_testfinal/checkpoint-best/ \
			--model_name_or_path ./model_testfinal/checkpoint-best/ \
			--pred_model_dir ./model_testfinal/checkpoint-best/ \
			--test_result_dir ./model_testfinal/test_results_staqc
```

To test on the webQuery test set, you run the following command:

```shell
python run_classifier.py \
			--model_type roberta \
			--task_name staqc \
			--do_predict \
			--test_file test_search.txt \
			--max_seq_length 200 \
			--per_gpu_eval_batch_size 2 \
			--data_dir ./data/ \
			--output_dir ./model_testfinal/checkpoint-best/ \
			--model_name_or_path ./model_testfinal/checkpoint-best/ \
			--pred_model_dir ./model_testfinal/checkpoint-best/ \
			--test_result_dir ./model_testfinal/test_results_webquery
```

## Results

The results on CodeSearch-webQuery are shown as below:

|    testset    |  model   | Precision | Recall |  F1   | Accuracy |
| :-----------: | :------: | :-------: | :----: | :---: | :------: |
|  test-StaQC   | Code-HNN |   0.770   | 0.859  | 0.812 |   N/A    |
|  test-StaQC   | RoBERTa  |   0.812   | 0.785  | 0.798 |  0.813   |
|  test-StaQC   | CodeBERT |   0.839   | 0.803  | 0.821 |  0.827   |
| test-webQuery | RoBERTa  |   0.415   | 0.689  | 0.518 |  0.483   |
| test-webQuery | CodeBERT |   0.438   | 0.822  | 0.572 |  0.504   |

