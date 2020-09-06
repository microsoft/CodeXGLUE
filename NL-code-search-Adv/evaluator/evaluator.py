# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys,json
import numpy as np

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['url']]=js['idx']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            predictions[js['url']]=js['answers']
    return predictions

def calculate_scores(answers,predictions):
    scores=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        flag=False
        for rank,idx in enumerate(predictions[key]):
            if idx==answers[key]:
                scores.append(1/(rank+1))
                flag=True
                break
        if flag is False:
            scores.append(0)
    result={}
    result['MRR']=round(np.mean(scores),4)
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for POJ-104 dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    print(scores)

if __name__ == '__main__':
    main()