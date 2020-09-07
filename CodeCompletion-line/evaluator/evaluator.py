# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import os
import logging
import argparse
from fuzzywuzzy import fuzz
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels, in json format.")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    preds = open(args.predictions, "r").readlines()
    gts = open(args.answers, "r").readlines()

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = 0.0
    edit_sim = 0.0
    for pred, gt in zip(preds, gts):
        pred = pred.strip()
        gt = json.loads(gt)["gt"]
        edit_sim += fuzz.ratio(pred, gt)
        if pred.split() == gt.split():
            EM += 1

    logger.info(f"Edit sim: {round(edit_sim/total, 2)}, EM: {round(EM/total*100, 2)}")

if __name__ == "__main__":
    main()
