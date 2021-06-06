import torch
import json
import numpy as np
from torch import tensor
from torch._C import long
from .Basic import BasicFormatter

class NewsRecFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.item2id = json.load(open(config.get("data", "item2id")))

        self.features = ["os", "country", "region", "ref_type"]
        self.label2id = json.load(open(config.get("data", "label2id"), "r"))

    def encode_item_seq(self, iseq, max_len=None):
        embs = [i["vec"] for i in iseq]
        category = [self.label2id["label2id"]["category"][i["category"]] for i in iseq]
        if max_len is not None and len(embs) < max_len:
            embs += [np.zeros(250)] * (max_len - len(embs))
            category += [self.label2id["label2id"]["category"]["PAD"]] * (max_len - len(embs))
        return embs, category

    def process(self, data, config, mode, *args, **params):
        # news feature
        news_emb = []
        news_category = []

        # click feature
        click_feature = [[] for i in range(len(self.features))]

        # cand_feature
        cand_emb = []
        cand_category = []

        inpseqs = []

        inpmask = []

        for user in data:
            if len(user["seq"]) > self.max_len:
                seq = user["seq"][-self.max_len:]
            else:
                seq = user["seq"]

            inpseq = seq[:-1]
            # 点击的特征序列
            for fid, feat in enumerate(self.features):
                fseq = [self.label2id["label2id"][feat][click["feat"][feat]] for click in inpseq]
                if len(fseq) < self.max_len:
                    fseq += [self.label2id["label2id"][feat]["PAD"]] * (self.max_len - len(fseq))
                click_feature[fid].append(fseq)

            # 输入的itemid序列
            iseq = [self.item2id[item["iid"]] for item in inpseq]
            if len(iseq) < self.max_len:
                iseq += [self.item2id["PAD"]] * (self.max_len - len(iseq))
            inpmask.append([1] * len(iseq) + [0] * (self.max_len - len(iseq)))
            inpseqs.append(iseq)

            # 输入的新闻序列
            iemb, icate = self.encode_item_seq([i["item_feature"] for i in inpseq])
            news_emb.append(iemb)
            news_category.append(icate)

            # 候选新闻的特征序列
            golden = seq[-1]
            allcand = [golden["item_feature"]] + user["negs"]
            cemb, ccate = self.encode_item_seq(allcand)
            cand_emb.append(cemb)
            cand_category.append(ccate)

        ret = {
            "inpseq": torch.tensor(inpseqs, dtype=torch.long),
            "clickf": torch.tensor(click_feature, dtype=torch.long),
            "news_emb": torch.tensor(news_emb, dtype=torch.float),
            "news_category": torch.tensor(news_category, dtype=torch.long),
            "cand_emb": torch.tensor(cand_emb),
            "cand_category": torch.tensor(cand_category, dtype=torch.long),
            "inpmask": torch.tensor(inpmask, dtype=torch.long),
            "labelmask": torch.tensor([user["mask"] for user in data], dtype=torch.long)
        }

        return ret
