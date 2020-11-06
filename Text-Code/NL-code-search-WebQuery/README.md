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

Here we present NL-code-search-WebQuery dataset,  a  testing  set  of  Python code  search of 1,046  query-code pairs with code search intent and their human annotations. The realworld user queries are collected from Bing query logs and the code for queries are from CodeSearchNet. You can find our testing set in `./data/test_webquery.json` .

Since there's no direct training set for our WebQueryTest set, we finetune the models on an external training set by using the documentation-function pairs in the training set o fCodeSearchNet AdvTest as positive instances. For each documentation, we also randomly sample 31 more functions to form negative instances. You can run the following command to download and preprocess the data:

```shell
cd data
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip
python preprocess.py
rm -r python
rm -r *.pkl
rm python.zip
cd ..
```

#### Data statistics

Data statistics of NL-code-search-WebQuery are shown in the table below:

|               | #Examples |
| ------------- | :-------: |
| test_webquery |   1,046   |


## Fine-tuning

You can use the following command to finetune:

```shell
python code/run_classifier.py \
			--model_type roberta \
			--task_name webquery \
			--do_train \
			--do_eval \
			--eval_all_checkpoints \
			--train_file train_codesearchnet_31.json \
			--dev_file dev_codesearch_net.json \
			--max_seq_length 200 \
			--per_gpu_train_batch_size 32 \
			--per_gpu_eval_batch_size 32 \
			--learning_rate 1e-5 \
			--num_train_epochs 3 \
			--gradient_accumulation_steps 1  \
			--warmup_steps 5000 \
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
| test-WebQuery | RoBERTa  |   49.50   | 70.62  | 58.20 |  58.64   |
| test-WebQuery | CodeBERT |   49.92   | 75.12  | 59.98 |  59.56   |

## Cite

If you use this code or our NL-code-search-WebQuery dataset, please considering citing CodeXGLUE:	

<pre><code>@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}</code>
</pre>


