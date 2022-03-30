# coding=utf-8
# Copyright 2022 Kevin Jesse.
# Licensed under the CC-by license.
# Lint as: python3

import logging
import sys

import numpy as np


# load predictions
def load_type_file(path):
    guesses = {}  # keeping to codexglue for potential change to index based eval
    with open(path) as file:
        for line in file:
            enum_pred = line.strip().split("\t")
            guesses[int(enum_pred[0])] = enum_pred[1]
    # return [v for _, v in sorted(guesses.items(), key=lambda item: item[0])] 
    return guesses


def calculate_scores(answers, predictions):
    Acc = []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key] == predictions[key])

    scores = {}
    scores['Acc'] = np.mean(Acc)
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate leaderboard predictions for Type Prediction TypeScript dataset.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in .txt format.")
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in .txt format.")

    args = parser.parse_args()
    answers = load_type_file(args.answers)
    predictions = load_type_file(args.predictions)
    scores = calculate_scores(answers, predictions)
    print(scores)


if __name__ == "__main__":
    main()
