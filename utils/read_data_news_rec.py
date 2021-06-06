import csv
import os

from numpy import index_exp

data_path = "/data/disk1/private/xcj/BigDataClass/data/NewsRec"
train_log = open(os.path.join(data_path, "train_click_log.csv"), "r")
reader = csv.reader(train_log)
id2seq = {}
rows = list(reader)
item2num = {}
user2env = {}
user2group = {}

os2num = {}
country2num = {}
region2num = {}
ref2num = {}

item2id = {"PAD": 0, "CLS": 1}
for r in rows[1:]:
    uid, iid, time = r[0], r[1], r[2]
    item2id[iid] = len(item2id)

    if uid not in id2seq:
        id2seq[uid] = []
        user2env[uid] = set()
    id2seq[uid].append((r[1], r[2]))
    user2env[uid].add(r[3])
    if iid not in item2num:
        item2num[iid] = 0
    item2num[iid] += 1

    if r[-4] not in os2num:
        os2num[r[-4]] = 0
    os2num[r[-4]] += 1
    if r[-3] not in country2num:
        country2num[r[-3]] = 0
    country2num[r[-3]] += 1
    if r[-2] not in region2num:
        region2num[r[-2]] = 0
    region2num[r[-2]] += 1
    if r[-1] not in ref2num:
        ref2num[r[-1]] = 0
    ref2num[r[-1]] += 1
print(country2num)
print(len(region2num), region2num)
print(len(ref2num), ref2num)
print(len(os2num), os2num)
# from IPython import embed; embed()


category2num = {}
articles = open(os.path.join(data_path, "articles.csv"), "r")
reader = csv.reader(articles)
head_row = next(reader)
for row in reader:
    if row[1] not in category2num:
        category2num[row[1]] = 0
    category2num[row[1]] += 1
print(len(category2num), category2num)

import json
fout = open("item2id.json", "w")
print(json.dumps(item2id, indent = 2), file = fout)
fout.close()
