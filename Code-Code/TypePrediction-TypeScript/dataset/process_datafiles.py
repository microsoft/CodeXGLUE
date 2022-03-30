#!/usr/bin/env python3
# Copyright 2022 Kevin Jesse.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import getopt
# run like ./process_datafiles.py -v 50000
import json
import os
import sys
from collections import Counter


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def calculate_vocab(all_training_labels, vocab_size):
    c = Counter(all_training_labels)
    vocab = ['UNK'] + list(list(zip(*c.most_common(len(c))))[0])
    tag2id = {tag: id for id, tag in enumerate(vocab, 0)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    truncated_vocab = list(tag2id.keys())[:vocab_size + 1]
    with open('vocab_{}.txt'.format(int(vocab_size)), 'w') as filehandle:
        for listitem in truncated_vocab:
            filehandle.write('%s\n' % listitem)
    return truncated_vocab


# ../vocab_50k.txt
def load_vocab(vocab_file):
    with open(vocab_file, 'r') as file:
        vocab = file.readlines()
        vocab = [line.rstrip() for line in vocab]
        return vocab


def create_dataframe(datafile, vocabulary, test_or_valid=False):
    data_for_frame = []
    input_ids = []
    labels_ids = []
    for row in datafile:
        for f_name, file in row['filedata'].items():
            if not file['annotations']:
                continue
            simplified_annotations = [annot['ty'] for annot in file['annotations']]

            for i in range(len(simplified_annotations)):
                simplified_annotations[i]['category'] = file['annotations'][i]['category']
            if test_or_valid:
                label_map = {annotation['loc']: annotation['type'] for annotation in simplified_annotations if
                             annotation['category'] == 'UserAnnot'}
            else:
                label_map = {annotation['loc']: annotation['type'] for annotation in simplified_annotations}
            labels = [label_map[i] if i in label_map else None for i in range(len(file['tokens']))]
            n_labels = []
            for label in labels:
                n_label = label
                if label is None:
                    n_labels.append(n_label)
                    continue

                if label not in vocabulary:
                    n_label = 'UNK'
                elif label == 'any':
                    n_label = None
                n_labels.append(n_label)

            if all(v is None for v in n_labels):
                continue

            data_row = {'tokens': file['tokens'], 'labels': n_labels, 'url': row['url'], 'path': f_name,
                        'commit_hash': row['commit_hash'], 'file': os.path.basename(f_name)}
            data_for_frame.append(data_row)
    return data_for_frame


def main(vocab_size):
    train_datafile0 = load_jsonl('ManyTypes4TypeScript_zenodo/datafiles/splits/train_datafile0.jsonl')
    train_datafile1 = load_jsonl('ManyTypes4TypeScript_zenodo/datafiles/splits/train_datafile1.jsonl')
    train_datafile2 = load_jsonl('ManyTypes4TypeScript_zenodo/datafiles/splits/train_datafile2.jsonl')
    test_datafile = load_jsonl('ManyTypes4TypeScript_zenodo/datafiles/splits/test_datafile.jsonl')
    valid_datafile = load_jsonl('ManyTypes4TypeScript_zenodo/datafiles/splits/valid_datafile.jsonl')

    all_training_labels = {}
    for row in train_datafile0 + train_datafile1 + train_datafile2:
        for file in row['filedata'].values():
            if not file['annotations']:
                continue
            simplified_annotations = [annot['ty'] for annot in file['annotations']]
            for ty in simplified_annotations:
                if ty['type'] in all_training_labels:
                    all_training_labels[ty['type']] += 1
                else:
                    all_training_labels[ty['type']] = 1

    vocab = calculate_vocab(all_training_labels, vocab_size)

    train_dataset0 = create_dataframe(train_datafile0, vocab)
    dump_jsonl(train_dataset0, 'train0.jsonl')
    train_dataset1 = create_dataframe(train_datafile1, vocab)
    dump_jsonl(train_dataset1, 'train1.jsonl')
    train_dataset2 = create_dataframe(train_datafile2, vocab)
    dump_jsonl(train_dataset2, 'train2.jsonl')

    test_dataset = create_dataframe(test_datafile, vocab, test_or_valid=True)  # skips Inferred type locations
    valid_dataset = create_dataframe(valid_datafile, vocab, test_or_valid=True)
    dump_jsonl(test_dataset, 'test.jsonl')
    dump_jsonl(valid_dataset, 'valid.jsonl')
    return


if __name__ == "__main__":
    vocab_size = None
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hv:", ["vocab_size="])
    except getopt.GetoptError:
        print('process_datafiles.py -v <vocab_size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('process_datafiles.py -v <vocab_size>')
            sys.exit()
        elif opt in ("-v", "--vocab_size"):
            vocab_size = int(arg)

    if not vocab_size:
        # if len(argv) and int(argv[-1]) != vocab_size:
        print('process_datafiles.py -v <vocab_size>')
        sys.exit(2)

    main(vocab_size)
