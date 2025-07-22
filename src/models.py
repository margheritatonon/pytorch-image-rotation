import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models #for resnet

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


#TRANSFORMER MODEL
#idea: break the image into patches and then treat the patches of the image like a sequence and process them
#we attend to the different patches of the image at the same time
#so we have diff components: input embedding to turn image into vector, position encoding to give info abt position, multi head to pay attention to diff parts at once, feedforward layers
#adapted from the PyTorch implementation
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, embedding_dimension):
        super().__init__()
        self.patch_size = patch_size
        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        self.embedding_dimension = embedding_dimension
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_chans, embedding_dimension, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # [B, D, H', W']
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        return x

class RotationTransformer(nn.Module):
    """
    Outputs 2 numbers per image: (sin θ, cos θ).
    """

    def __init__(self, image_size, patch_size, in_chans, embedding_dimension, depth, number_heads, mlp_ratio, dropout, activation="relu", unit_norm=True):
        super().__init__()

        #Patch + position embeddings
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_chans, embedding_dimension)
        self.num_patches = self.patch_embedding.num_patches
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dimension))
        self.pos_drop = nn.Dropout(dropout)

        #Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=number_heads,
            dim_feedforward=int(embedding_dimension * mlp_ratio),
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        #Head
        self.norm = nn.LayerNorm(embedding_dimension)
        self.head = nn.Linear(embedding_dimension, 2)   # (sin, cos)
        self.unit_norm = unit_norm

        #initialization of parameters
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):                            # x: (B, 6, H, W)
        x = self.patch_embedding(x)                  # (B, N, D)
        x = x + self.position_embedding
        x = self.pos_drop(x)

        x = self.encoder(x)                          # (B, N, D)
        x = self.norm(x)
        x = x.mean(dim=1)                            # Global average pool

        rot_vec = self.head(x)                       # (B, 2)
        if self.unit_norm:
            rot_vec = F.normalize(rot_vec, dim=-1)   # project to unit circle
        return rot_vec  


#TODO: confused because we want the output to be sine and cosine since we are predicting the angle, but im not sure how to do that here.
#not sure the approach is correct

def build_transformer_model(config):
    """
    Function to get the Transformer model for image rotation prediction.
    Used in main.py
    """
    return RotationTransformer(
        image_size=config["model"].get("image_size", 32),
        patch_size=config["model"].get("patch_size", 4),
        in_chans=config["model"].get("in_chans", 6),
        embedding_dimension=config["model"].get("embedding_dimension", 128),
        depth=config["model"].get("depth", 4),
        number_heads=config["model"].get("number_heads", 8),
        mlp_ratio=config["model"].get("mlp_ratio", 4.0),
        dropout=config["model"].get("dropout", 0.1), #these are the parameters that we can tune for the transformer model
        activation=config["model"].get("activation", "relu"),
        unit_norm=config["model"].get("unit_norm", True)  # whether to normalize
    )


# RESNET
class ResNet(nn.Module):
    def __init__(self, unit_norm = True):
        super().__init__()
        self.unit_norm = unit_norm
        self.backbone = models.resnet18(pretrained=True)

        #modifying the first layer to accept 6 channels bc of the images we have
        self.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        #modifying the last layer to output 2 values (sin, cos)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
    
    def forward(self, x):
        rot = self.backbone(x)
        if self.unit_norm:
            rot = F.normalize(rot, dim=-1)
        return rot

def build_resnet_model(config):
    """
    Function to get the ResNet model for image rotation prediction.
    Used in main.py
    """
    return ResNet(unit_norm=config["model"].get("unit_norm", True))  # whether to normalize the output

def get_model(config):
    if config["model"]["type"] == "custom":
        return build_custom_model(config)
    elif config["model"]["type"] == "siamese":
        return build_siamese_model(config)
    elif config["model"]["type"] == "transformer":
        return build_transformer_model(config)
    elif config["model"]["type"] == "resnet":
        return build_resnet_model(config)
    #can add more models here later - create functions that build the models based on different types/architectures
