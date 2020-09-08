# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys

from bleu import _bleu

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for BigCloneBench dataset.')
    parser.add_argument('--references', '-ref',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-pre',help="filename of the leaderboard predictions, in txt format.")
    
    args = parser.parse_args()

    refs = [x.strip() for x in open(args.references, 'r', encoding='utf-8').readlines()]
    pres = [x.strip() for x in open(args.predictions, 'r', encoding='utf-8').readlines()]
    
    assert len(refs) == len(pres)

    length = len(refs)
    count = 0
    for i in range(length):
    	r = refs[i]
    	p = pres[i]
    	if r == p:
    		count += 1
    acc = round(count/length*100, 2)
    
    bleu_score = round(_bleu(args.references, args.predictions),2)
    
    print('BLEU:',  bleu_score, '; Acc:', acc)
    
if __name__ == '__main__':
    main()
