from .ImgModel import ImgCNN, ImgMLP
from .Bert4Rec import Bert4Rec
model_list = {
    "ImgCNN": ImgCNN,
    "ImgMLP": ImgMLP,
    "BERT4Rec": Bert4Rec
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
