"""Dinov2 feature extractor."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2VitFeatureExtractor(nn.Module):
    """DINO V2 Vision Transformer Feature Extractor."""
    def __init__(self, model_name='dinov2_vitb14_reg'):
        super(DinoV2VitFeatureExtractor, self).__init__()
        dino_v2_models = {
            "dinov2_vits14_reg": (14, 384), #* patch_size, output dims
            "dinov2_vitb14_reg": (14, 768),
            "dinov2_vitl14_reg": (14, 1024),
            "dinov2_vitb14": (14, 768),
            "dinov2_vitl14": (14, 1024),
            "dinov2_vitg14": (14, 1536),
        }
        patch_size, embed_dim = dino_v2_models[model_name]
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        assert self.model.embed_dim == embed_dim
        self.embed_dim = self.model.embed_dim
        self.patch_size = patch_size

    def forward(self, x):
        _, _, height, width = x.size()
        #* check image dims divisible by patch_size
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        
        feats = self.model.get_intermediate_layers(x, reshape=True)[0] #* batch_size, num_patches, self.embed_dim
        return feats