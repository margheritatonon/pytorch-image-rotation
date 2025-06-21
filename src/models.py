import torch.nn as nn

#class RotationModel(nn.Module):
#    def __init__(self, )

def build_custom_model(config):
    """
    Function to get the models for image rotation prediction.
    Used in main.py
    """
    layers = []
    for layer in config["model"]["layers"]: #looping through the layers in the config
        if "flatten" in layer and layer["flatten"] == True:
            layers.append(nn.Flatten())
        elif "linear" in layer: #if the layer is a linear layer:
            in_dim, out_dim = layer["linear"] #taking the dimensions of the layer
            layers.append(nn.Linear(in_dim, out_dim))
        elif "activation" in layer:
            activation = layer["activation"]
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "softmax":
                layers.append(nn.Softmax(dim=1))
        #can add more stuff here later if needed

    model = nn.Sequential(*layers) #creating the model
    return model

def get_model(config):
    if config["model"]["type"] == "custom":
        return build_custom_model(config)
    #can add more models here later - create functions that build the models based on different types/architectures
