# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys, json, os
import numpy as np
import argparse


def read_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            answers[line.split('\t')[0]] = line.split('\t')[1]
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            predictions[line.split('\t')[0]] = line.split('\t')[1]
    return predictions


def calculate_scores(answers, predictions):
    scores = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        a = answers[key]
        p = predictions[key]
        scores.append(a==p)
    result = sum(scores) / len(scores)
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for ClozeTest-maxmin dataset.')
    parser.add_argument('--answers_staqc', '-as', help="filename of the labels on staqc test set, in txt format.")
    parser.add_argument('--predictions_staqc', '-ps', help="filename  of the leaderboard predictions on staqc test set, in txt format.")
    parser.add_argument('--answers_webquery', '-aw', help="filename of the labels on staqc test set, in txt format.")
    parser.add_argument('--predictions_webquery', '-pw', help="filename  of the leaderboard predictions on staqc test set, in txt format.")
    args = parser.parse_args()

    answers = read_answers(args.answers_staqc)
    predictions = read_predictions(args.predictions_staqc)
    acc_staqc = calculate_scores(answers, predictions)
    print('NL-code-search-WebQuery on staqc test set, acc: {}'.format(acc_staqc))

    answers = read_answers(args.answers_webquery)
    predictions = read_predictions(args.predictions_webquery)
    acc_webquery = calculate_scores(answers, predictions)
    print('NL-code-search-WebQuery on WebQuery test set, acc: {}'.format(acc_webquery))


if __name__ == '__main__':
    main()