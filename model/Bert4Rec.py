import torch
import torch.nn as nn
import torch.nn.functional as F
import json
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
        self.click_feature = nn.Linear(self.feature_dim * len(self.features), self.hidden_size)

        self.category_emb = nn.Embedding(462, self.feature_dim)
        self.news_feature = nn.Linear(self.feature_dim + 250, self.hidden_size)
        self.newsemb_norm = nn.BatchNorm1d(250)

        self.input_feature = nn.Linear(self.hidden_size * 3, self.hidden_size)

        self.output = nn.Linear(self.hidden_size * 2, 1)
        # self.output = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, config, gpu_list, acc_result, mode):
        '''
        "inpseq": batch, seq_len
        "clickf": batch, fnum, seq_len
        "news_emb": batch, seq_len, 250
        "news_category": batch, seq_len
        "cand_emb": batch, cand_num, 250
        "cand_category": batch, cand_num
        "inpmask": batch, seq_len
        "labelmask": batch, cand_num
        '''

        cand_num = data["cand_emb"].shape[1]
        click_emb = torch.cat([self.os_emb(data["clickf"][:,0]),
                            self.country_emb(data["clickf"][:,1]),
                            self.region_emb(data["clickf"][:,2]),
                            self.ref_emb(data["clickf"][:,3])], dim = 2) # batch, seq_len, fnum * feature_dim
        clickf = self.click_feature(click_emb) # batch, seq_len, self.hidden_size
        
        news_emb = torch.transpose(self.newsemb_norm(torch.transpose(data["news_emb"], 1, 2)), 1, 2) # batch, seq_len, 250
        news_cate = self.category_emb(data["news_category"]) # batch, seq_len, feature_dim
        newsf = self.news_feature(torch.cat([news_emb, news_cate], dim = 2)) # batch, seq_len, self.hidden_size

        cand_emb = torch.transpose(self.newsemb_norm(torch.transpose(data["cand_emb"], 1, 2)), 1, 2)
        cand_cate = self.category_emb(data["cand_category"]) # batch, cand_num, feature_dim
        candf = self.news_feature(torch.cat([cand_emb, cand_cate], dim = 2)) # batch, cand_num, self.hidden_size

        item_emb = self.bert.get_input_embeddings()(data["inpseq"]) # batch, seq_len, self.hidden_size
        inp = self.input_feature(torch.cat([item_emb, newsf, clickf], dim = 2))

        output = self.bert(attention_mask=data["inpmask"], inputs_embeds=inp)
        hiddens = output["last_hidden_state"]
        feature = torch.max(hiddens, dim = 1) # batch, self.hidden_size

        score = self.output(torch.cat([feature.unsqueeze(1).repeat(1, cand_num, 1), candf], dim = 2)).squeeze(2) # batch, cand_num
        score = score - 100 * (1 - data["labelmask"])
        
        loss = self.criterion(score, torch.zeros(score.shape[0]).to(score.device))
        acc_result = acc(score, acc_result)

        return {'loss': loss, 'acc_result': acc_result}

def acc(score, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result["total"] += int(score.shape[0])
    acc_result["right"] += int((predict == 0).int().sum())
    return acc_result
