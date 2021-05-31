import torch.optim as optim
from transformers import AdamW
import torch
def get_params_for_prompt_optimization(module: torch.nn.Module):
    # params = [{"params": [], "lr": 1e-3}, {"params": [], "lr": 1e-5}]
    params = []
    for t in module.named_modules():
        if "prompt" in t[0]:
            # params[0]["params"].extend([p for p in list(t[1]._parameters.values()) if p is not None])
            params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
        # else:
        #     params[1]["params"].extend([p for p in list(t[1]._parameters.values()) if p is not None])
        #     # params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})

    return params


def init_optimizer(model, config, *args, **params):
    optimizer_type = config.get("train", "optimizer")
    learning_rate = config.getfloat("train", "learning_rate")
    if config.getboolean("prompt", "prompt_tune"):
        param_group = get_params_for_prompt_optimization(model)
        print("the number of params is ", len(param_group), [p.shape for ps in param_group for p in ps["params"]])
    else:
        param_group = model.parameters()
    if optimizer_type == "adam":
        optimizer = optim.Adam(param_group, lr=learning_rate,
                               weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(param_group, lr=learning_rate,
                              weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "adamw":
        optimizer = AdamW(param_group, lr=learning_rate,
                             weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    return optimizer