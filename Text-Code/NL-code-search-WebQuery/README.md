# CodeXGLUE -- Code Search (WebQueryTest)

**Update 2021.07.30:** We add CoSQA into CodeXGLUE, and recommend you to use this dataset as the training and development set of CodeXGLUE -- Code Search (WebQueryTest) Challenge. The dataset can be found in `./cosqa`. For more details about the dataset collection and usage, please refer to the [ACL 2021 paper](https://arxiv.org/abs/2105.13239) and the [GitHub repo](https://github.com/Jun-jie-Huang/CoCLR). 

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

Since there's no direct training set for our WebQueryTest dataset, we use the CoSQA dataset as the training set and dev set. CoSQA includes 20,604 labels for pairs of natural language queries and Python codes, with almost the same collections and data format as the 1,046 pairs in WebQueryTest. You can find the CoSQA training and dev set in `./data/cosqa_train.json` and   `./data/cosqa_dev.json` . The detailed construction of CoSQA can be found in the paper [CoSQA: 20,000+ Web Queries for Code Search and Question Answering (In Proceedings of ACL 2021)]().

#### Data statistics

Data statistics of WebQueryTest are shown in the table below:

|              | #Examples |
| :----------: | :-------: |
| WebQueryTest |   1,046   |

Data statistics of CoSQA are shown in the table below:

|                | #Examples |
| :------------: | :-------: |
| CoSQA-training |  20,000   |
|   CoSQA-dev    |    604    |

## Fine-tuning

You can use the following command to finetune:

```shell
python code/run_classifier.py \
			--model_type roberta \
			--do_train \
			--do_eval \
			--eval_all_checkpoints \
			--train_file cosqa_train.json \
			--dev_file cosqa_dev.json \
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

@article{CoSQA,
  title={[CoSQA: 20,000+ Web Queries for Code Search and Question Answering},
  journal={arXiv},
  year={2021},
}

@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}
```



