# CodeXGLUE -- Code Refinement

Here is the pipeline for the code refinement task.


## Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0


## Data

Data statistics of this dataset are shown in the below table:

|         | #Examples | #Examples |
|         |   Small   |   Medium  |
| ------- | :-------: | :-------: |
|  Train  |   46,680  |   52,364  |
|  Valid  |    5,835  |    6,545  |
|   Test  |    5,835  |    6,545  |


## Run
Just move to the "code" folder, and choose the pipline you want to run. Model1&2 mean
Roberta(Code) and CoderBERT respectively. Model3 means GraphCoderBERT. The pretrained 
model is in the "CoderBERT" subfolder in the respective pipline. To do training and 
test, please refer to the according run-xx.sh file. 


