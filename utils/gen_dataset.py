from genericpath import isdir
import json
from posixpath import join
import random
import os

path = "/data/disk1/private/xcj/BigDataClass/data/omniglot_resized"
imgsys = [os.path.join(path, fname) for fname in os.listdir(path) if os.path.isdir(os.path.join(path, fname))]
imgc = [os.path.join(p, cha) for p in imgsys for cha in os.listdir(p) if os.path.isdir(os.path.join(p, cha))]


random.seed(1102)
categories = random.sample(imgc, 50)
train_data = []
test_data = []
label2id = {}
for p in categories:
    fnames = [os.path.join(p, name) for name in os.listdir(p) if name[-3:] == "png"]
    random.shuffle(fnames)
    train_data.extend(fnames[:5])
    test_data.extend(fnames[15:])

    labelname = set([name.split("/")[-1].split("_")[0] for name in fnames])
    assert len(labelname) == 1
    label2id[list(labelname)[0]] = len(label2id)


print("train data num: ", len(train_data), "\ttest data num: ", len(test_data))
outpath = "/data/disk1/private/xcj/BigDataClass/data/data_for_train/ver1_5n"
fout = open(os.path.join(outpath, "train.json"), "w")
print(json.dumps(train_data, indent=2), file=fout)
fout.close()

fout = open(os.path.join(outpath, "test.json"), "w")
print(json.dumps(test_data, indent=2), file=fout)
fout.close()

fout = open(os.path.join(outpath, "label2id.json"), "w")
print(json.dumps(label2id, indent=2), file=fout)
fout.close()