import os
import random

# 训练集与验证集的划分比例
trainval_percent = 0.4
train_percent = 0.6
xmlfilepath = "/Annotations"  # 标签文件
# txtsavepath = "/JPEGImages"  # 图片文件
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftest = open("/test.txt", "w")  # 生成的test.txt路径
ftrain = open("/train.txt", "w")  # 生成的train.txt路径

for i in list:
    name = total_xml[i][:-4] + "\n"
    if i in trainval:
        ftest.write(name)
    else:
        ftrain.write(name)

ftrain.close()
ftest.close()
