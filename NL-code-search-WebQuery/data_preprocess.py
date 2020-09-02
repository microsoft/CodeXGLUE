import pickle
import os

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

def generate_my_format(data, quests, qc_merge):
    new_data = []
    for data in data:
        label = data[7]
        q = quests[data[0][0]]
        c = qc_merge[data[0][0]][data[0][1]]
        new_data.append({'title':q,
                         'code':c,
                         'label':label})
    return new_data

my_train_data = generate_my_format(train_data, questions, questions_code_merge)
my_dev_data = generate_my_format(dev_data, questions, questions_code_merge)
my_test_data = generate_my_format(test_data, questions, questions_code_merge)

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string

def write(path, data):
    with open(path, 'w', encoding='utf-8') as fp:
        for instance in data:
            # fp.write(str(instance['label']) +
            #          '<CODESPLIT>' +
            #          .strip() +
            #          '<CODESPLIT>' +
            #           +
            #          '\n')
            example = (str(instance['label']), instance['title'].strip(), ' '.join([format_str(token) for token in instance['code'].split(' ')]))
            example = '<CODESPLIT>'.join(example)
            fp.write(example+'\n')

write('./data/train.txt', my_train_data)
write('./data/dev.txt', my_dev_data)
write('./data/test_staqc.txt', my_test_data)