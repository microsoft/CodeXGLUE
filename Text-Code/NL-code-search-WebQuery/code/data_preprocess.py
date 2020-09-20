# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pickle
import os
import json

pickle_path = './StackOverflow-Question-Code-Dataset/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle'
code_snippet = pickle.load(open(pickle_path, 'rb'))
pickle_path = './StackOverflow-Question-Code-Dataset/annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle'
questions = pickle.load(open(pickle_path, 'rb'))
questions_key = set(questions.keys())
print(len(questions_key))
code_question_key = set([inst[0] for inst in code_snippet.keys()])
print(len(code_question_key))
questions_code_merge = {k: {} for k in questions_key}
for (q_id, c_id), c in code_snippet.items():
    questions_code_merge[q_id][c_id] = c
print(len(questions_code_merge))

path = './StackOverflow-Question-Code-Dataset/data/data_hnn/python/train/data_partialcontext_shared_text_vocab_in_buckets.pickle'
train_data = pickle.load(open(path, 'rb'))  # 2932
path = './StackOverflow-Question-Code-Dataset/data/data_hnn/python/valid/data_partialcontext_shared_text_vocab_in_buckets.pickle'
dev_data = pickle.load(open(path, 'rb'))  # 976
path = './StackOverflow-Question-Code-Dataset/data/data_hnn/python/test/data_partialcontext_shared_text_vocab_in_buckets.pickle'
test_data = pickle.load(open(path, 'rb'))  # 976
train_data = [i for inst in train_data for i in inst]
dev_data = [i for inst in dev_data for i in inst]
test_data = [i for inst in test_data for i in inst]


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string

def generate_my_format(data, type, quests, qc_merge):
    new_data = []
    for index, inst in enumerate(data):
        idx = 'staqc-'+type+'-'+str(index+1)
        label = inst[7]
        q = quests[inst[0][0]]
        c = qc_merge[inst[0][0]][inst[0][1]]
        # c = ' '.join([format_str(token) for token in c.split(' ')])
        if type == 'test':
            new_data.append({'idx': idx,
                             'doc': q,
                             'code': c})
        else:
            new_data.append({'idx':idx,
                             'doc':q,
                             'code':c,
                             'label':label})
    return new_data

my_train_data = generate_my_format(train_data, 'train', questions, questions_code_merge)
my_dev_data = generate_my_format(dev_data, 'dev', questions, questions_code_merge)
my_test_data = generate_my_format(test_data, 'test', questions, questions_code_merge)


def write(path, data):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=1)

write('./data/train_staqc.json', my_train_data)
write('./data/dev_staqc.json', my_dev_data)
write('./data/test_staqc.json', my_test_data)