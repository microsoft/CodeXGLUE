import json
import os
language='python'

train=[]
for root, dirs, files in os.walk(language+'/final'):
    for file in files:
        temp=os.path.join(root,file)
        if '.jsonl' in temp:
            if 'train' in temp:
                train.append(temp)

data={}                    
for file in train:
    if '.gz' in file:
        os.system("gzip -d {}".format(file))
        file=file.replace('.gz','')
    with open(file) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            data[js['url']]=js
cont=0                    
with open('train.jsonl','w') as f, open("train.txt") as f1:
    for line in f1:
        line=line.strip()
        js=data[line].copy()
        js['idx']=cont
        cont+=1
        f.write(json.dumps(js)+'\n')

            
data={}                    
with open('test_code.jsonl') as f:
    for line in f:
        line=line.strip()
        js=json.loads(line)
        data[js['url']]=js

                    
with open('valid.jsonl','w') as f, open("valid.txt") as f1:
    for line in f1:
        line=line.strip()
        js=data[line].copy()
        js['idx']=cont
        cont+=1
        f.write(json.dumps(js)+'\n')
            
with open('test.jsonl','w') as f, open("test.txt") as f1:
    for line in f1:
        line=line.strip()
        js=data[line].copy()
        js['idx']=cont
        cont+=1
        f.write(json.dumps(js)+'\n')