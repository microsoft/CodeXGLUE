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
            answers[line.split('<CODESPLIT>')[0]] = line.split('<CODESPLIT>')[1]
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            predictions[line.split('<CODESPLIT>')[0]] = line.split('<CODESPLIT>')[1]
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
    parser.add_argument('--answers', '-a', help="directory name of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', help="directory name  of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    for lang in ['ruby', 'javascript', 'go', 'python', 'java', 'php']:
        answers = read_answers(os.path.join(args.answers, lang, 'answers.txt'))
        predictions = read_predictions(os.path.join(args.predictions, lang, 'predictions.txt'))
        acc = calculate_scores(answers, predictions)
        print('ClozeTest-maxmin:{}, acc: {}'.format(lang, acc))


if __name__ == '__main__':
    main()