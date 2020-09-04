# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import json

def extract_answers(filename):
	cluster={}
	with open(filename) as f:
		for line in f:
			line=line.strip()
			js=json.loads(line)
			if js['label'] not in cluster:
				cluster[js['label']]=set()
			cluster[js['label']].add(js['index'])
	answers=[]
	for key in cluster:
		for idx1 in cluster[key]:
			temp={}
			temp['index']=idx1
			temp['answers']=[]
			for idx2 in cluster[key]:
				if idx1!=idx2:
					temp['answers'].append(idx2)
			answers.append(temp)
	return answers


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract answers from code files.')
    parser.add_argument('--codefile', '-c',help="filename of the code examples.")
    parser.add_argument('--outfile', '-o',help="filename of output.")
    args = parser.parse_args()
    answers=extract_answers(args.codefile)
    with open(args.outfile,'w') as f:
    	for line in answers:
    		f.write(json.dumps(line)+'\n')

if __name__ == '__main__':
    main()