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

Since there's no direct training set for our WebQueryTest dataset, we suggest using two external training sets: 1. CodeSearchNet; 2. CoSQA.

1. To finetune the models on CodeSearchNet, we provide scripts to obtain the documentation-function pairs in the training set o CodeSearchNet AdvTest as positive instances. For each documentation, we also randomly sample 7 more functions to form negative instances. The following command is used to download and preprocess the data:

```shell
cd data
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip
python preprocess.py
rm -r python
rm -r *.pkl
rm python.zip
cd ..
```

2. You can also continue finetuning the model on CoSQA dataset. CoSQA includes 20,604 labels for pairs of natural language queries and Python codes, with almost the same collections and data format as the 1,046 pairs in WebQueryTest. You can find the CoSQA training and dev set in `./data/cosqa_train.json` and   `./data/cosqa_dev.json` . The detailed construction of CoSQA can be found in the paper [CoSQA: 20,000+ Web Queries for Code Search and Question Answering (In Proceedings of ACL 2021)]().

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

You can use the following command to finetune the model on CodeSearchNet:

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
			--warmup_steps 1000 \
			--evaluate_during_training \
			--data_dir ./data/ \
			--output_dir ./model_codesearchnet \
			--encoder_name_or_path microsoft/codebert-base 
```

You can then use the following command to continue finetuning the model on CoSQA:

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
			--output_dir ./model_cosqa_continue_training \
			--encoder_name_or_path ./model_codesearchnet 

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
			--output_dir ./model_cosqa_continue_training/checkpoint-best-aver/ \
			--encoder_name_or_path microsoft/codebert-base \
			--pred_model_dir ./model_cosqa_continue_training/checkpoint-last/ \
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

|       dataset        |  model   | Accuracy |
| :------------------: | :------: | :------: |
|    CodeSearchNet     | RoBERTa  |  40.92   |
|    CodeSearchNet     | CodeBERT |  47.80   |
| CodeSearchNet+ CoSQA | CodeBERT |  52.87   |

## Cite

If you use this code or our WebQueryTest dataset, please considering citing CodeXGLUE and CodeSearchNet:	

```
@article{Lu2021CodeXGLUEAM,
  title={CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation},
  author={Shuai Lu and Daya Guo and Shuo Ren and Junjie Huang and Alexey Svyatkovskiy and Ambrosio Blanco and Colin Clement and Dawn Drain and Daxin Jiang and Duyu Tang and Ge Li and Lidong Zhou and Linjun Shou and Long Zhou and Michele Tufano and Ming Gong and Ming Zhou and Nan Duan and Neel Sundaresan and Shao Kun Deng and Shengyu Fu and Shujie Liu},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.04664}
}

@inproceedings{Huang2020CoSQA20,
  title={CoSQA: 20, 000+ Web Queries for Code Search and Question Answering},
  author={Junjie Huang and Duyu Tang and Linjun Shou and Ming Gong and Ke Xu and Daxin Jiang and Ming Zhou and Nan Duan},
  booktitle={ACL/IJCNLP},
  year={2020}
}

@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}
```



