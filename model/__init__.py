from .ImgModel import ImgCNN, ImgMLP
model_list = {
    "ImgCNN": ImgCNN,
    "ImgMLP": ImgMLP,
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
