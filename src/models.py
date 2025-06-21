import torch.nn as nn
import torch

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

#SIAMESE: we first define the Siamese model class
class SiameseModel(nn.Module):
    def __init__(self, embedding_net, comparison_net):
        super().__init__()
        self.embedding_net = embedding_net
        self.comparison_net = comparison_net
    
    def forward(self, input1, input2):
        emb1 = self.embedding_net(input1) 
        emb2 = self.embedding_net(input2)
        combined = torch.cat((emb1, emb2), dim=1) #combining the embeddings
        output = self.comparison_net(combined)
        return output.squeeze()

def build_siamese_model(config):
    """
    Function to get the Siamese model for image rotation prediction.
    Used in main.py
    """
    embedding_net = build_custom_model({"model": {"layers": config["model"]["embedding_layers"]}})
    comparison_net = build_custom_model({"model": {"layers": config["model"]["comparison_layers"]}})
    return SiameseModel(embedding_net, comparison_net)

def get_model(config):
    if config["model"]["type"] == "custom":
        return build_custom_model(config)
    elif config["model"]["type"] == "siamese":
        return build_siamese_model(config)
    #can add more models here later - create functions that build the models based on different types/architectures
