# CodeXGLUE -- Method Generation

Here is the introduction and pipeline for method generation task.

## Task Definition

Method generation is the prediction of a method body implementation conditioned on a signature, a docstring, and any more context. 

## Dataset

We use CodeSearchNet Python dataset. The CodeSearchNet repositories are re-downloaded to extract all the methods, including their signatures, docstrings and bodies. We remove the methods that don't have docstrings and whose name contains 'test'. We preserve the context around this method for auxiliary information since it is really a difficult task to generator the method body only based on its signature/docstring. We also apply literal normalization for better user experience.

To download the preprocessed dataset, navigate to `dataset` directory, and run
```shell
git lfs clone https://huggingface.co/datasets/microsoft/codexglue_method_generation
```

## Difference from CONCODE
This dataset is a real-world code generation task for Python. Unlike CONCODE, in this dataset, function/variable names are not anonymized. We don't provide the class enviroment in a pre-defined way but provide all the context around the method to be generated. Besides, the method could be very long. The average tokens of each method is 89.0 while in CONCODE is 26.3.

### Data Format

The data format of each line in `train/dev/test.jsonl` is:
```json
{
    "signature": "def do_transform(self, v=<NUM_LIT:1>):",
    "body": "if not self.transform:<EOL><INDENT>return<EOL><DEDENT>try:<EOL><INDENT>self.latest_value = utils.Transform ...",
    "docstring": "Apply the transformation (if it exists) to the latest_value",
    "id": "f19:c4:m1"
}
```
The `id` indicts where you can find this method in the raw data. In this instance, it means the 2nd method in the 2nd class in the 19th file. We apply literal normalization to function signature and body, replace `\n` with `<EOL>` and keep track in INDENT and DEDENT.

As the original code could be auxiliary information for generation, in the `raw` directory, `train/dev/test.jsonl` saves the orginal code, in which each line is like:
```json
{
    "relative_path": "...",
    "original_string": "...",
    "file_hash": "6ab80abab195ef2aa7a179a083f620fc",
    "file_docstring": "",
    "methods": [
        {
            "attributes": {},
            "syntax_pass": true,
            "default_arguments": {},
            "original_string": "...",
            "byte_span": [112,242],
            "start_point": [6,0],
            "end_point": [10,20],
            "name": "pairwise",
            "signature": "def pairwise(iterable):",
            "body": "    a, b = tee(iterable)\n    next(b, None)\n    return zip(a, b)",
            "docstring": "s -> (s0,s1), (s1,s2), (s2, s3), ...",
            "id": "f3:m0"
        },
        ...
    ],
    "classes": [
        {
            "attributes": {},
            "class_docstring": "",
            "methods": [...],
            "byte_span": [46,704],
            "start_point": [5,0],
            "end_point": [26,44],
            "original_string": "...",
            "name": "TestPokeAPI",
            "id": "f3:c0"
        }
    ],
    "url": "https://github.com/mamikonyana/cryptotools",
    "repo_name": "mamikonyana/cryptotools",
    "license": {},
    "id": "f3"
}
```
The `id` is coresponding to the preprocessed dataset. You could use this information to get all the information you need. For example, the `id` of the method in preprocessed data is `f3:c0:m1`, then find the file with `f3` id in raw data, then find the class in `c0` id, finally the method with `m1` id. The `byte_span` attribute can be used for locating where the method is in the whole file. After that, you will get the code context around this method.

### Data Statistics

Data statistics are shown in the below table.

| Data Split  |  #Instances |
| ----------- | :---------: |
|    Train    |   893,538   |
|     Dev     |    20,000   |
|    Test     |    20,000   |



## Evaluator

We provide a script to evaluate predictions for this task, and report accuracy score. You can run the script like this:

```bash
python evaluator/evaluator.py -a=evaluator/answers.txt -p=evaluator/predictions.txt
```

Each line in the *.txt file is an output.

## Pipeline

We provides our implementation to fine-tune CodeGPT on method generation task.

### Dependency

- python 3.6 or 3.7
- torch>=1.4.0
- transformers>=3.3.0
- fuzzywuzzy

### Fine-tune
To fine-tune CodeGPT on method generation dataset in multi-GPU on a single machine, navigate to `code` directory, run:

```Shell
DATADIR=../dataset/codexglue_method_generation
OUTPUTDIR=../save
PRETRAINDIR=microsoft/CodeGPT-small-py-adaptedGPT2
LITFILE=../dataset/literals.json
LOGFILE=train.log
PER_NODE_GPU=${1} && echo PER_NODE_GPU: ${PER_NODE_GPU}         # set with your GPU number

python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run.py \
        --data_dir=$DATADIR \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --lit_file=$LITFILE \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=1 \
        --num_train_epochs=30 \
        --logging_steps=100 \
        --save_steps=10000 \
        --warmup_steps=10000 \
        --overwrite_output_dir \
        --seed=42
```


### Evaluation && Inference

It's recommanded to run evaluation on single GPU. The predictions will be saved at `$OUTPUTDIR/test.output`

```Shell
GPU_ID=${1} && echo GPU_ID: ${GPU_ID}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
DATADIR=../dataset/codexglue_method_generation
OUTPUTDIR=../save/predictions
PRETRAINDIR=../save/checkpoint              # your model path
LITFILE=../dataset/literals.json
LOGFILE=eval.log

python -u run.py \
        --data_dir=$DATADIR \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --lit_file=$LITFILE \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_infer \
        --node_index 0 \
        --gpu_per_node 1 \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=1 \
        --num_train_epochs=30 \
        --logging_steps=100 \
        --save_steps=10000 \
        --warmup_steps=10000 \
        --overwrite_output_dir \
        --seed=42
```

## Result

TODO: We will add more baselines results on this task.

| Model                                       |    BLEU    |  Edit similarity  |
| ------------------------------------------- | :--------: | :---------------: |
| CodeGPT-adapted                             |  **10.14** |    **46.77**      |

## Reference

If you use method generation dataset, please also cite the following papers **in addition to our CodeXGLUE**:

<pre><code>@article{clement2021long,
  title={Long-Range Modeling of Source Code Files with eWASH: Extended Window Access by Syntax Hierarchy},
  author={Clement, Colin B and Lu, Shuai and Liu, Xiaoyu and Tufano, Michele and Drain, Dawn and Duan, Nan and Sundaresan, Neel and Svyatkovskiy, Alexey},
  journal={arXiv preprint arXiv:2109.08780},
  year={2021}
}</code></pre>

