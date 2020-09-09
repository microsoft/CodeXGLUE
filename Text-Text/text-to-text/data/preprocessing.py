

import sys
import codecs
import os


def com1(f1, f2, file1, file2, tag1, tag2):

    for l1, l2 in zip(file1, file2):
        #print (l1)
        #print (l2)
        f1.write(tag1 + " " + l1)
        f2.write(l2)
    
    for l1, l2 in zip(file1, file2):
        f1.write(tag2 + " " + l2)
        f2.write(l1)



def train_dev_test(f1, f2, tag):

    list_ = [[tag+"/da-en."+tag+".en", tag+"/da-en."+tag+".da", "2da", "2en"],
             [tag+"/no-en."+tag+".en", tag+"/no-en."+tag+".no", "2no", "2en"],
             [tag+"/lv-en."+tag+".en", tag+"/lv-en."+tag+".lv", "2lv", "2en"],
             [tag+"/zh-en."+tag+".en", tag+"/zh-en."+tag+".zh", "2zh", "2en"]]
    for l in list_:
        file1_ = codecs.open(l[0], "r", "utf-8")
        file2_ = codecs.open(l[1], "r", "utf-8")
        file1 = [line for line in file1_.readlines()]
        file2 = [line for line in file2_.readlines()]
        com1(f1, f2, file1, file2, l[2], l[3])

if __name__ == "__main__":
    list_ = ["train", "dev", "test"]
    if not os.path.exists("./processed"):
        os.mkdir("./processed")
    for l in list_:
        f1 = codecs.open("./processed/"+l+".all.src", "w", "utf-8")
        f2 = codecs.open("./processed/"+l+".all.tgt", "w", "utf-8")
        train_dev_test(f1, f2, l)


