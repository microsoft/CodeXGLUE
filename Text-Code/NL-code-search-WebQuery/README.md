# CodeXGLUE -- NL-code-search-WebQuery

Here is the code and data for NL-code-search-WebQuery task.

## Task Description

Code Search is aimed to find a code snippet which best matches the demand of the query. This task can be formulated in two scenarios: retrieval scenario and text-code classification scenario. In NL-code-search-WebQuery, we present the Code Search in text-code classification scenario.

In NL-code-search-WebQuery, a trained model needs to judge whether a code snippet answers a given natural language query, which can be formulated into a binary classification problem. 

Most  existing  code search datasets use code documentations or questions from online communities for software developers as queries, which is still different from real user search queries.  Therefore we provide NL-code-searhc-WebQuery  testing set.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.5.0


## Data

Here we present NL-code-searhc-WebQuery dataset,  a  testing  set  of  real  code  search  for Python of 1,046  query-code pairs with code search intent and their human annotations. The real user queries are collected from Bing query logs and the code for queries are from CodeSearchNet. You can find our testing set in `./data/test_webquery.json` .

Since there's no direct training set for our WebQueryTest set, we can finetune a pre-trained model in a zero-shot setting. The training and validation sets of NL-code-search-WebQuery data are collected from the StaQC python dataset, where each instance contains a StackOverflow title, a python code snippet and a 0/1 annotation of whether the code answers the title.

#### Download and Preprocess

For original StaQC dataset, you can download it in [this repo](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset) or you can run the following command: 

```shell
git clone https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset.git
python code/data_preprocess.py 
```

#### Data statistics

Data statistics of NL-code-search-WebQuery are shown in the table below:

|               | #Examples |
| ------------- | :-------: |
| test_webquery |   1,046   |


## Fine-tuning

You can use the following command to finetune a pre-trained model on the StaQC training set:

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

After generate predictions for WebQuery testset, you can use our provided script to evaluate:

```shell
python evaluator/evaluator.py \
		--answers_webquery evaluator/webquery_answers.txt \
		--predictions_webquery evaluator/webquery_predictions.txt
```

## Results

The results on NL-code-search-WebQuery are shown as below:

|    testset    |  model   | Precision | Recall |  F1   | Accuracy |
| :-----------: | :------: | :-------: | :----: | :---: | :------: |
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

