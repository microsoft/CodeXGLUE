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
			answers[js['index']]=js['answers']
	return answers

def read_predictions(filename):
	predictions={}
	with open(filename) as f:
		for line in f:
			line=line.strip()
			js=json.loads(line)
			predictions[js['index']]=js['answers']
	return predictions

def calculate_scores(answers,predictions):
	scores=[]
	for key in answers:
		if key not in predictions:
				logging.error("Missing prediction for index {}.".format(key))
				sys.exit()
		a=set(answers[key])
		p=set(predictions[key])
		if len(a)!=len(p):
				logging.error("Mismatch the number of answers for index {}.".format(key))
				sys.exit()		
		scores.append(len(set(a&p))/len(a))
	result={}
	result['MAP']=round(np.mean(scores),4)
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