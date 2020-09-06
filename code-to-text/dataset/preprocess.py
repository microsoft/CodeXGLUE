import json
import os

for language in ['ruby','go','java','javascript','php','python']:
    print(language)
    train,valid,test=[],[],[]
    for root, dirs, files in os.walk(language+'/final'):
        for file in files:
            temp=os.path.join(root,file)
            if '.jsonl' in temp:
                if 'train' in temp:
                    train.append(temp)
                elif 'valid' in temp:
                    valid.append(temp)
                elif 'test' in temp:
                    test.append(temp)   
                    
    train_data,valid_data,test_data={},{},{}
    for files,data in [[train,train_data],[valid,valid_data],[test,test_data]]:
            for file in files:
                if '.gz' in file:
                    os.system("gzip -d {}".format(file))
                    file=file.replace('.gz','')
                with open(file,encoding='latin-1') as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data[js['url']]=js
    for tag,data in [['train',train_data],['valid',valid_data],['test',test_data]]:
        with open('{}/{}.jsonl'.format(language,tag),'w') as f, open("{}/{}.txt".format(language,tag),encoding='latin-1') as f1:
            for line in f1:
                line=line.strip()
                if line in data:
                    f.write(json.dumps(data[line])+'\n')
