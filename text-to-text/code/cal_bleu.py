# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from __future__ import absolute_import
import os
import sys
import codecs
from bleu import _bleu
import numpy as np

def cal_bleu(hyp, ref):
    
    dev_bleu = round(_bleu(ref, hyp), 2)
    f1 = codecs.open(ref, "r", "utf-8")
    f2 = codecs.open(hyp, "r", "utf-8")
    accs = []
    for l1, l2 in zip(f1.readlines(), f2.readlines()):
        accs.append(l1.strip()==l2.strip())
    
    print ("bleu-4: ", str(dev_bleu))
    #print ("xMatch: ", str(round(np.mean(accs)*100,4)))
    
if __name__ == "__main__":
    hyp = sys.argv[1]
    ref = sys.argv[2]
    cal_bleu(hyp, ref)

