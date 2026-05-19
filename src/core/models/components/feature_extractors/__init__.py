

import torch.nn as nn
import timm

from .dinov2_features import DinoV2VitFeatureExtractor
from .rein import ReinsDinoV2, ReinsDinoV2Custom, EVA2, ReinsEVA2

__all__ = [
    "DinoV2VitFeatureExtractor",
    "ReinsDinoV2",
    "ReinsDinoV2Custom",
    "EVA2",
    "ReinsEVA2"
]

def build_feature_extractor(backbone_name, 
                            backbone_cfg=dict(),
                            ckpt=None,
                            layers=None,
                            pretrained=True) -> nn.Module:
    """Build the feature extractor for normalizing flow.
    Args:
        backbone_name (str): Backbone name.
        ckpt (str, optional): Checkpoint path. Defaults to None.
        layers (list[int], optional): List of layers to extract features from. Defaults to None.
        pretrained (bool, optional): Boolean to check whether to use a pre-trained backbone from ImageNet.
    Returns:
        nn.Module: Feature extractor.
    """
    if backbone_name in ("resnet18", "wide_resnet50_2", "wide_resnet101_2"):
        # if not pre_trained:
        #     encoder = ResNetStudentFeatureExtractor(backbone_name, ckpt, layers)
        #     channels, scales = encoder.channels, encoder.scales
        # else:
        encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=layers,
            pretrained_cfg_overlay=dict(file=ckpt) if ckpt is not None else None)
        channels = encoder.feature_info.channels()
        scales = encoder.feature_info.reduction()
    elif backbone_name in ("dino_resnet50"):
        encoder = DinoResNetFeatureExtractor(layers=layers)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name in ("dino_vitb8", "dino_vitb16"):
        encoder = DinoVitFeatureExtractor(model_name=backbone_name)
        channels = [encoder.embed_dim]
        scales = [encoder.patch_size]
    elif backbone_name in ("dinov2_vitb14", 
                           "dinov2_vitl14", 
                           "dinov2_vitg14",
                           "dinov2_vitb14_reg",
                           "dinov2_vitl14_reg"):
        encoder = DinoV2VitFeatureExtractor(model_name=backbone_name)
        channels = [encoder.embed_dim]
        scales = [encoder.patch_size]
    elif backbone_name.lower() == 'deeplabv3':
        encoder = DeepLabV3FeatureExtractor(ckpt=ckpt,
                                            layers=layers)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name.lower() in ['mit-b3', 'segformer']:
        encoder_cls = {'mit-b3': MiTFeatureExtractor, 'segformer': SegFormerFeatureExtractor}
        encoder = encoder_cls[backbone_name.lower()](ckpt=ckpt, layers=layers)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name.lower() == 'segwide':
        encoder = SegWideFeatureExtractor(ckpt=ckpt, layers=layers)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name.lower() == 'mae':
        encoder = MAEFeatureExtractor(pretrained=pretrained, pretrain_ckpt=ckpt)
        channels = [encoder.embed_dim]
        scales = [encoder.patch_size]
    elif backbone_name.lower() == 'dav2':
        encoder = DepthAnythingV2(encoder_size='l', ckpt=ckpt)
        channels = [encoder.embed_dim]
        scales = [encoder.patch_size]
    elif backbone_name.lower() == 'dino-dav2':
        encoder = DinoDaV2FeatureExtractor(ckpt=ckpt)
        channels = [encoder.embed_dim]
        scales = [encoder.patch_size]
    elif backbone_name.lower() in ['internimage_b', 'internimage_l']:
        encoder = FlashInternImageFeatureExtractor(model=backbone_name, 
                                                   layers=layers, 
                                                   ckpt=ckpt,
                                                   pretrained=pretrained)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name.lower() == 'internimage':
        encoder = InternImageFeatureExtractor(model='internimage', layers=layers, ckpt=ckpt)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name.lower() == 'internimage_upernet':
        encoder = InternImageUPerNetFeatureExtractor(layers=layers, ckpt=ckpt)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name.lower() == 'swin':
        encoder = SwinFeatureExtractor(layers=layers, ckpt=ckpt, pretrained=pretrained)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name.lower() == 'swin_d2':
        encoder = SwinD2FeatureExtractor(layers=layers, 
                                         ckpt=ckpt, 
                                         pretrained=pretrained,
                                         **backbone_cfg)
        channels = encoder.channels
        scales = encoder.scales
    elif backbone_name.lower() == 'rein_dinov2':
        encoder = ReinsDinoV2(model='l',ckpt=ckpt)
        channels = [encoder.embed_dim]
        scales = [encoder.patch_size]
    elif backbone_name.lower() == 'rein_dinov2b':
        encoder = ReinsDinoV2(model='b', ckpt=ckpt)
        channels = [encoder.embed_dim]
        scales = [encoder.patch_size]
    elif backbone_name.lower() == 'rein_dinov2_custom':
        encoder = ReinsDinoV2Custom(ckpt=ckpt, **backbone_cfg)
        channels = [encoder.embed_dim]
        scales = [encoder.patch_size]
    elif backbone_name.lower() == 'eva02':
        encoder = EVA2(pretrained=ckpt)
        channels = [encoder.num_features]
        scales = [encoder.patch_size]
    elif backbone_name.lower() == 'rein_eva02':
        encoder = ReinsEVA2(ckpt=ckpt)
        channels = [encoder.num_features]
        scales = [encoder.patch_size]
    else:
        msg = f"Backbone {backbone_name} is not supported."
        raise ValueError(msg)
    
    return encoder, channels, scales