# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import sys

path = "./saved_models/multi_model/"

f1 = open(path+"test_1.gold", "r")
f1_ = open(path+"test_1.output", "r")

f2 = open(path+"test_1.en-da.da.gold", "w")
f2_ = open(path+"test_1.en-da.da.output", "w")

f3 = open(path+"test_1.en-da.en.gold", "w")
f3_ = open(path+"test_1.en-da.en.output", "w")

f4 = open(path+"test_1.en-no.no.gold", "w")
f4_ = open(path+"test_1.en-no.no.output", "w")

f5 = open(path+"test_1.en-no.en.gold", "w")
f5_ = open(path+"test_1.en-no.en.output", "w")

f6 = open(path+"test_1.en-lv.lv.gold", "w")
f6_ = open(path+"test_1.en-lv.lv.output", "w")

f7 = open(path+"test_1.en-lv.en.gold", "w")
f7_ = open(path+"test_1.en-lv.en.output", "w")


f8 = open(path+"test_1.en-zh.zh.gold", "w")
f8_ = open(path+"test_1.en-zh.zh.output", "w")

f9 = open(path+"test_1.en-zh.en.gold", "w")
f9_ = open(path+"test_1.en-zh.en.output", "w")


i = 0

for l1, l2 in zip(f1.readlines(), f1_.readlines()):

	if i<1000:
		f2.write(l1)
		f2_.write(l2)
	elif i<2000:
		f3.write(l1)
		f3_.write(l2)
	elif i<3000:
		f4.write(l1)
		f4_.write(l2)
	elif i<4000:
		f5.write(l1)
		f5_.write(l2)
	elif i<5000:
		f6.write(l1)
		f6_.write(l2)
	elif i<6000:
		f7.write(l1)
		f7_.write(l2)
	elif i<7000:
		f8.write(l1)
		f8_.write(l2)
	else:
		f9.write(l1)
		f9_.write(l2)
	i += 1
