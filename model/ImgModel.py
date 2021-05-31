import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class ImgCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ImgCNN, self).__init__()
        self.label2id = json.load(open(config.get("data", "label2id"), "r"))
        kernel_size = 7
        pad = (kernel_size - 1) // 2
        self.featureExtractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = kernel_size, stride = 1, padding = pad),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size = kernel_size, stride = 1, padding = pad),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(32, 64, kernel_size = kernel_size, stride = 1, padding = pad),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = kernel_size, stride = 1, padding = pad),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.25),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p = 0.25),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p = 0.25),
            nn.Linear(512, len(self.label2id)),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, config, gpu_list, acc_result, mode):
        out = self.featureExtractor(data["inputx"].unsqueeze(1))
        score = self.classifier(out.view(out.size()[0], -1))

        loss = self.criterion(score, data["label"])
        acc_result = acc(score, data['label'], acc_result)

        return {'loss': loss, 'acc_result': acc_result}


class ImgMLP(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ImgMLP, self).__init__()
        self.label2id = json.load(open(config.get("data", "label2id"), "r"))

        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.25),
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p = 0.25),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p = 0.25),
            nn.Linear(512, len(self.label2id)),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, config, gpu_list, acc_result, mode):
        # out = self.featureExtractor(data["inputx"].unsqueeze(1))
        score = self.classifier(data["inputx"].view(data["inputx"].size()[0], -1))

        loss = self.criterion(score, data["label"])
        acc_result = acc(score, data['label'], acc_result)

        return {'loss': loss, 'acc_result': acc_result}



def acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())
    return acc_result