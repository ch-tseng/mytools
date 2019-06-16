import random
import glob, os
import os.path

#---------------------------------------------------------
testRatio = 0.2
cfg_folder = 'cfg-diabnext_300-277'
ds_imglist = 'filelist.list'

fileList = []
outputTrainFile = cfg_folder + "/train.txt"
outputTestFile = cfg_folder + "/test.txt"


f = open( os.path.join(cfg_folder, ds_imglist), "r")

with open(os.path.join(cfg_folder, ds_imglist)) as fp:  
    for cnt, line in enumerate(fp):
        #print("Line {}: {}".format(cnt, line))
        fileList.append(line.replace('\n',''))

fp.close()

print("total image files: ", len(fileList))

testCount = int(len(fileList) * testRatio)
trainCount = len(fileList) - testCount

a = range(len(fileList))
test_data = random.sample(a, testCount)
#train_data = random.sample(a, trainCount)
train_data = [x for x in a if x not in test_data]

print ("Train:{} images".format(len(train_data)))
print("Test:{} images".format(len(test_data)))

with open(outputTrainFile, 'a') as the_file:
    for i in train_data:
        the_file.write(fileList[i] + "\n")

the_file.close()

with open(outputTestFile, 'a') as the_file:
    for i in test_data:
        the_file.write(fileList[i] + "\n")

