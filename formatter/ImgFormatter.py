import torch
import json
import numpy as np
from .Basic import BasicFormatter


class ImgFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.label2id = json.load(open(config.get("data", "label2id"), "r"))

    def process(self, data, config, mode, *args, **params):
        inputx = []
        label = []
        for ins in data:
            inputx.append(1 - ins["img"] / 255.0)
            label.append(ins["label"])

        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

        return ret
