# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import os
import logging
import argparse
from fuzzywuzzy import fuzz
import json
import re
from bleu import _bleu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def post_process(code):
    code = code.replace("<EOL>", "\n").replace("<INDENT>", " ").replace("<DEDENT>", " ")
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return " ".join(code.split())

def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    preds = open(args.predictions, "r").readlines()
    gts = open(args.answers, "r").readlines()

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    edit_sim = 0.0
    for pred, gt in zip(preds, gts):
        pred = post_process(pred.strip())
        gt = post_process(gt.strip())
        edit_sim += fuzz.ratio(pred, gt)

    bleu_score = round(_bleu(args.answers, args.predictions), 2)
    logger.info(f"Edit sim: {round(edit_sim/total, 2)}, BLEU: {bleu_score}")

if __name__ == "__main__":
    main()
