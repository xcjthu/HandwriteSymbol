import json
from torch.utils.data import Dataset
import numpy as np
import csv
import os
import random

class NewsRecDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode

        self.data_path = config.get("data", "data_path")
        train_log = open(os.path.join(self.data_path, "train_click_log.csv"), "r")
        reader = csv.reader(train_log)
        rows = list(reader)
        self.id2seq = {}
        itemids = set()
        for r in rows[1:]:
            uid, iid, time = r[0], r[1], r[2]
            if uid not in self.id2seq:
                self.id2seq[uid] = []
            self.id2seq[uid].append({"iid": r[1], "time": r[2], "env": r[3], "os": r[5], "country": r[6], "region": r[7], "ref_type": r[8]})
            itemids.add(iid)
        for uid in self.id2seq:
            self.id2seq[uid].sort(key = lambda x:x["time"])

        self.item2feature = {}
        itememb = open(os.path.join(self.data_path, "articles_emb.csv"), "r")
        ireader = csv.reader(itememb)
        header_row = next(ireader)
        for row in ireader:
            if row[0] in itemids:
                vec = np.array([float(v) for v in row[1:]])
                self.item2feature[row[0]] = {"vec": vec}
        itemf = open(os.path.join(self.data_path, "articles.csv"), "r")
        ifreader = csv.reader(itemf)
        header_row = next(ifreader)
        for row in ifreader:
            if row[0] in itemids:
                self.item2feature[row[0]]["category"] = row[1]
                self.item2feature[row[0]]["word_count"] = row[3]

        del_data = []
        for uid in self.id2seq:
            if len(self.id2seq) <= 2:
                del_data.append(uid)
                continue
            for cid, click in enumerate(self.id2seq[uid]):
                self.id2seq[uid][cid]["item_feature"] = self.item2feature[click["iid"]]
        for uid in del_data:
            self.id2seq.pop(uid)
        self.all_itemid = list(itemids)

        if mode == "train":
            for uid in self.id2seq:
                self.id2seq[uid] = self.id2seq[uid][:-1]
        self.data = list(self.id2seq.values())
        print("==" * 10, mode, "==" * 10)
        print("user num", len(self.data))
        print("item num", len(itemids))


    def __getitem__(self, item):
        iseq = self.data[item]
        gitem = set([i["iid"] for i in iseq])
        negs = set(random.sample(self.all_itemid, 99))
        mask = [1] + [0 if nid in gitem else 1 for nid in negs]
        negs = [self.item2feature[nid] for nid in negs]
        return {"seq": iseq, "negs": negs, "mask": mask}

    def __len__(self):
        return len(self.data)

