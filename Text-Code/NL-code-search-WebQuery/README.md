# CodeXGLUE -- NL-code-search-WebQuery

Here is the code and data for NL-code-search-WebQuery task.

## Task Description

Code Search is aimed to find a code snippet which best matches the demand of the query. This task can be formulated in two scenarios: retrieval scenario and text-code classification scenario. In NL-code-search-WebQuery, we present the Code Search in text-code classification scenario.

In NL-code-search-WebQuery, a trained model needs to judge whether a code snippet answers a given natural language query, which can be formulated into a binary classification problem. 

Here we present a test set where natural language queries come from Bing query log is created to test on real user queries. We also include the annotated StaQC dataset, a question-code dataset from StackOverflow.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.5.0


## Data

The training and validation sets of NL-code-search-WebQuery data are collected from StaQC, where each instance contains a StackOverflow title, a code snippet and a 0/1 annotation of whether the code answers the title.

As for test set, we include the original test set of StaQC. Also to apply to a wider scenario, we create a new test set where each instance contains a Bing Search query, a python code function and a 0/1 annotation of whether the code answers the query. The newly created test set is in `./data/test_webquery.json` .

#### Download and Preprocess

For original StaQC dataset, you can download it in [this repo](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset) or you can run the following command: 

```shell
git clone https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset.git
python code/data_preprocess.py 
```

#### Data statistics

Data statistics of NL-code-search-WebQuery are shown in the below table:

|               | #Examples |
| ------------- | :-------: |
| train_staqc   |   2,932   |
| dev_staqc     |    976    |
| test_staqc    |    976    |
| test_webquery |   1,046   |


## Fine-tuning

You can fine-tune a pre-trained model by the following command:

```shell
python code/run_classifier.py \
			--model_type roberta \
			--task_name staqc \
			--do_train \
			--do_eval \
			--eval_all_checkpoints \
			--train_file train_staqc.json \
			--dev_file dev_staqc.json \
			--max_seq_length 200 \
			--per_gpu_train_batch_size 16 \
			--per_gpu_eval_batch_size 16 \
			--learning_rate 1e-4 \
			--num_train_epochs 10 \
			--gradient_accumulation_steps 1  \
			--warmup_steps 180 \
			--overwrite_output_dir \
			--data_dir ./data/ \
			--output_dir ./model/ \
			--model_name_or_path microsoft/codebert-base \
			--config_name roberta-base
```

## Evaluation

To test on the StaQC test set, you run the following command. It will automatically generate predictions to `--prediction_file`.

```shell
python code/run_classifier.py \
			--model_type roberta \
			--task_name staqc \
			--do_predict \
			--test_file test_staqc.json \
			--max_seq_length 200 \
			--per_gpu_eval_batch_size 2 \
			--data_dir ./data/ \
			--output_dir ./model/checkpoint-best/ \
			--model_name_or_path ./model/checkpoint-best/ \
			--pred_model_dir ./model/checkpoint-best/ \
			--test_result_dir ./model/test_results_staqc \
			--prediction_file ./evaluator/staqc_predictions.txt			
```

To test on the WebQuery testset, you run the following command. Also it will automatically generate predictions to `--prediction_file`.

```shell
python code/run_classifier.py \
			--model_type roberta \
			--task_name webquery \
			--do_predict \
			--test_file test_webquery.json \
			--max_seq_length 200 \
			--per_gpu_eval_batch_size 2 \
			--data_dir ./data/ \
			--output_dir ./model/checkpoint-best/ \
			--model_name_or_path ./model/checkpoint-best/ \
			--pred_model_dir ./model/checkpoint-best/ \
			--test_result_dir ./model/test_results_webquery \
			--prediction_file ./evaluator/webquery_predictions.txt
```

After generate predictions for StaQC testset and WebQuery testset, you can use our provided script to evaluate:

```shell
python evaluator/evaluator.py \
		--answers_staqc evaluator/staqc_answers.txt \
		--predictions_staqc evaluator/staqc_predictions.txt \
		--answers_webquery evaluator/webquery_answers.txt \
		--predictions_webquery evaluator/webquery_predictions.txt
```

## Results

The results on NL-code-search-WebQuery are shown as below:

|    testset    |  model   | Precision | Recall |  F1   | Accuracy |
| :-----------: | :------: | :-------: | :----: | :---: | :------: |
|  test-StaQC   | Code-HNN |   0.770   | 0.859  | 0.812 |   N/A    |
|  test-StaQC   | RoBERTa  |   0.785   | 0.853  | 0.791 |  0.801   |
|  test-StaQC   | CodeBERT |   0.792   | 0.798  | 0.821 |  0.825   |
| test-WebQuery | RoBERTa  |   0.408   | 0.756  | 0.530 |  0.459   |
| test-WebQuery | CodeBERT |   0.422   | 0.910  | 0.576 |  0.460   |

## Cite

If you use this code or our NL-code-search-WebQuery dataset, please considering citing CodeXGLUE and StaQC:	

<pre><code>@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}</code>
</pre>
<pre>
<code>@inproceedings{yao2018staqc,
  title={StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow},
  author={Yao, Ziyu and Weld, Daniel S and Chen, Wei-Peng and Sun, Huan},
  booktitle={Proceedings of the 2018 World Wide Web Conference},
  pages={1693--1703},
  year={2018}
} </code>
</pre>

