# CodeXGLUE -- Code Search (WebQueryTest)

## Task Description

Code Search is aimed to find a code snippet which best matches the demand of the query. This task can be formulated in two scenarios: retrieval scenario and text-code classification scenario. In WebQueryTest , we present the Code Search in text-code classification scenario.

In WebQueryTest, a trained model needs to judge whether a code snippet answers a given natural language query, which can be formulated into a binary classification problem. 

Most  existing  code search datasets use code documentations or questions from online communities for software developers as queries, which is still different from real user search queries.  Therefore we provide WebQueryTest testing set.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.5.0


## Data

Here we present WebQueryTest dataset,  a  testing  set  of  Python code  search of 1,046  query-code pairs with code search intent and their human annotations. The realworld user queries are collected from Bing query logs and the code for queries are from CodeSearchNet. You can find our testing set in `./data/test_webquery.json` .

Since there's no direct training set for our WebQueryTest dataset, we finetune the models on an external training set by using the documentation-function pairs in the training set o CodeSearchNet AdvTest as positive instances. For each documentation, we also randomly sample 31 more functions to form negative instances. You can run the following command to download and preprocess the data:

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

Data statistics of WebQueryTest are shown in the table below:

|              | #Examples |
| :----------: | :-------: |
| WebQueryTest |   1,046   |


## Fine-tuning

You can use the following command to finetune:

```shell
python code/run_classifier.py \
			--model_type roberta \
			--do_train \
			--do_eval \
			--eval_all_checkpoints \
			--train_file train_codesearchnet_7.json \
			--dev_file dev_codesearchnet.json \
			--max_seq_length 200 \
			--per_gpu_train_batch_size 16 \
			--per_gpu_eval_batch_size 16 \
			--learning_rate 1e-5 \
			--num_train_epochs 3 \
			--gradient_accumulation_steps 1 \
			--warmup_steps 5000 \
			--evaluate_during_training \
			--data_dir ./data/ \
			--output_dir ./model \
			--encoder_name_or_path microsoft/codebert-base 

```

## Evaluation

To test on the WebQueryTest, you run the following command. Also it will automatically generate predictions to `--prediction_file`.

```shell
python code/run_classifier.py \
			--model_type roberta \
			--do_predict \
			--test_file test_webquery.json \
			--max_seq_length 200 \
			--per_gpu_eval_batch_size 2 \
			--data_dir ./data \
			--output_dir ./model/checkpoint-best-aver/ \
			--encoder_name_or_path microsoft/codebert-base \
			--pred_model_dir ./model/checkpoint-last/ \
			--prediction_file ./evaluator/webquery_predictions.txt 
			
```

After generate predictions for WebQueryTest, you can use our provided script to evaluate:

```shell
python evaluator/evaluator.py \
		--answers_webquery ./evaluator/webquery_answers.txt \
		--predictions_webquery evaluator/webquery_predictions.txt
```

## Results

The results on WebQueryTest are shown as below:

|   dataset    |  model   |  F1   | Accuracy |
| :----------: | :------: | :---: | :------: |
| WebQueryTest | RoBERTa  | 57.49 |  40.92   |
| WebQueryTest | CodeBERT | 58.95 |  47.80   |

## Cite

If you use this code or our WebQueryTest dataset, please considering citing CodeXGLUE and CodeSearchNet:	

```
@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}
```

```
@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}
```



