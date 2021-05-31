import json
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from torchvision import transforms
import numpy as np
class ImageDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)

        self.rotation = transforms.RandomRotation(20)
        data_list = json.load(open(self.data_path, "r"))
        self.data = []
        self.label2id = json.load(open(config.get("data", "label2id"), "r"))
        for path in data_list:
            lname = path.split("/")[-1].split("_")[0]
            img = Image.open(path).convert("RGB")
            self.data.append({"img": np.array(img)[:,:,0], "label": self.label2id[lname]})
            # if mode == "train":
            #     for i in range(3):
            #         img_tmp = self.rotation(img)
            #         self.data.append({"img": np.array(img_tmp)[:,:,0], "label": self.label2id[lname]})
           
        print(mode, "data num", len(self.data))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
