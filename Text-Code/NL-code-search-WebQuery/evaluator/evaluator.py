# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys, json, os
import numpy as np
import argparse
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def read_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            answers[line.split('\t')[0]] = int(line.split('\t')[1])
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            predictions[line.split('\t')[0]] = int(line.split('\t')[1])
    return predictions


def calculate_scores(answers, predictions):
    y_trues, y_preds = [], []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
    scores={}
    scores['Precision']=precision_score(y_trues, y_preds)
    scores['Recall']=recall_score(y_trues, y_preds)
    scores['F1']=f1_score(y_trues, y_preds)
    scores['Accuracy']=accuracy_score(y_trues, y_preds)
    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for ClozeTest-maxmin dataset.')
    parser.add_argument('--answers_webquery', '-aw', help="filename of the labels on webquery test set, in txt format.")
    parser.add_argument('--predictions_webquery', '-pw', help="filename  of the leaderboard predictions on webquery test set, in txt format.")
    args = parser.parse_args()

    answers = read_answers(args.answers_webquery)
    predictions = read_predictions(args.predictions_webquery)
    acc_webquery = calculate_scores(answers, predictions)
    # print('NL-code-search-WebQuery on WebQuery test set, acc: {}'.format(acc_webquery))
    print('NL-code-search-WebQuery on WebQuery test set:')
    print(acc_webquery)


if __name__ == '__main__':
    main()