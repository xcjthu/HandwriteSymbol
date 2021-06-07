from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.nn.modules.activation import ReLU
from transformers import BertModel, BertConfig

class Bert4Rec(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Bert4Rec, self).__init__()

        self.hidden_size = 128
        self.config = BertConfig(vocab_size = 31120, hidden_size = self.hidden_size, num_hidden_layers = 2, 
                    num_attention_heads = 2, intermediate_size = 256, max_position_embeddings = 20)
        self.bert = BertModel(self.config)

        self.features = ["os", "country", "region", "ref_type"]
        self.feature_dim = 32
        self.os_emb = nn.Embedding(9, self.feature_dim)
        self.country_emb = nn.Embedding(12, self.feature_dim)
        self.region_emb = nn.Embedding(29, self.feature_dim)
        self.ref_emb = nn.Embedding(8, self.feature_dim)
        # self.clickF = nn.MultiheadAttention(self.feature_dim, num_heads=2)
        self.click_feature = nn.Sequential(
            nn.Linear(self.feature_dim * len(self.features), self.hidden_size),
            nn.ReLU()
        )

        self.category_emb = nn.Embedding(462, self.feature_dim)
        self.news_feature = nn.Sequential(
            nn.Linear(self.feature_dim + 250, self.hidden_size),
            nn.ReLU()
        )
        self.news_feature2 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )
        self.newsemb_norm = nn.BatchNorm1d(250)
        self.newsf_norm = nn.BatchNorm1d(self.hidden_size)
        self.click_norm = nn.BatchNorm1d(self.feature_dim * len(self.features))

        self.input_feature = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )

        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self.gelu = nn.GELU()
        self.bias = nn.Embedding(31120, 1)
        # self.output = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, config, gpu_list, acc_result, mode):
        '''
        "inpseq": batch, seq_len
        "clickf": fnum, batch, seq_len
        "news_emb": batch, seq_len, 250
        "news_category": batch, seq_len
        "cand_emb": batch, cand_num, 250
        "cand_category": batch, cand_num
        "inpmask": batch, seq_len
        "labelmask": batch, cand_num
        '''

        cand_num = data["cand_emb"].shape[1]
        click_emb = torch.cat([self.os_emb(data["clickf"][0]),
                            self.country_emb(data["clickf"][1]),
                            self.region_emb(data["clickf"][2]),
                            self.ref_emb(data["clickf"][3])], dim = 2) # batch, seq_len, fnum * feature_dim
        click_emb = torch.transpose(self.click_norm(torch.transpose(click_emb, 1, 2)), 1, 2)
        clickf = self.click_feature(click_emb) # batch, seq_len, self.hidden_size

        news_emb = torch.transpose(self.newsemb_norm(torch.transpose(data["news_emb"], 1, 2)), 1, 2) # batch, seq_len, 250
        news_cate = self.category_emb(data["news_category"]) # batch, seq_len, feature_dim
        newsf = self.news_feature(torch.cat([news_emb, news_cate], dim = 2)) # batch, seq_len, self.hidden_size
        item_ids = self.bert.get_input_embeddings()(data["inpseq"]) # batch, seq_len, self.hidden_size
        newsf2 = self.news_feature2(torch.cat([item_ids, newsf], dim = 2))
        newsf2 = torch.transpose(self.newsf_norm(torch.transpose(newsf2, 1, 2)), 1, 2)

        cand_ids = self.bert.get_input_embeddings()(data["cand_ids"])
        cand_emb = torch.transpose(self.newsemb_norm(torch.transpose(data["cand_emb"], 1, 2)), 1, 2)
        cand_cate = self.category_emb(data["cand_category"]) # batch, cand_num, feature_dim
        candf = self.news_feature(torch.cat([cand_emb, cand_cate], dim = 2)) # batch, cand_num, self.hidden_size
        candf2 = self.news_feature2(torch.cat([cand_ids, candf], dim = 2))
        candf2 = torch.transpose(self.newsf_norm(torch.transpose(candf2, 1, 2)), 1, 2)

        inp = self.input_feature(torch.cat([newsf2, clickf], dim = 2))

        output = self.bert(attention_mask=data["inpmask"], inputs_embeds=inp)
        # output = self.bert(data["inpseq"], attention_mask=data["inpmask"])
        hiddens = output["last_hidden_state"] * data["inpmask"].unsqueeze(2) # batch, seq_len, self.hidden_size
        feature = torch.max(hiddens, dim = 1)[0] # batch, self.hidden_size

        # score = self.output(torch.cat([feature.unsqueeze(1).repeat(1, cand_num, 1), candf2], dim = 2)).squeeze(2) # batch, cand_num
        score = torch.bmm(self.gelu(self.output(feature)).unsqueeze(1), torch.transpose(candf2, 1, 2)).squeeze(1) + self.bias(data["cand_ids"]).squeeze(2) # batch, cand_num
        if mode == "test":
            return {"loss": 0, "output": list(zip(score.tolist(), data["cid"], data["uid"]))}
        score = score - 10000 * (1 - data["labelmask"])

        loss = self.criterion(score, torch.zeros(score.shape[0], dtype=torch.long).to(score.device))
        # acc_result = acc(score, acc_result)
        acc_result = IRMetric(score, acc_result)

        return {'loss': loss, 'acc_result': acc_result}

def acc(score, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result["total"] += int(score.shape[0])
    acc_result["right"] += int((predict == 0).int().sum())
    return acc_result

def IRMetric(score, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'P@1/NDCG@1': 0, 'mrr': 0.0, "NDCG@10": 0.0, "P@10": 0}
    rank = (-score).argsort().argsort()[:,0]
    acc_result["mrr"] += float((1 / (rank.float() + 1.0)).sum())
    acc_result["P@10"] += int((rank < 10).sum())
    acc_result["P@1/NDCG@1"] += int((rank == 0).sum())
    acc_result["NDCG@10"] += float((1 / torch.log2(rank.float() + 2.0)).sum())
    acc_result["total"] += rank.shape[0]
    return acc_result
