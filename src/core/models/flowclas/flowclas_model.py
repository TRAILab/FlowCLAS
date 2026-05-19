"""Torch module: frozen backbone, per-scale flows, projection and optional segmentation heads."""

import logging

import normflows as nf
import torch
import torch.nn.functional as F
from FrEIA.framework import SequenceINN
from FrEIA.modules import AffineCouplingOneSided, GLOWCouplingBlock, PermuteRandom
from torch import Tensor, nn
from mmengine.model import kaiming_init

from core.models.components.feature_extractors import build_feature_extractor
from core.models.components.flow import ActNorm, ActNormV2, build_subnet
from core.models.components.flow.distributions import MultivariateNormal


logger = logging.getLogger(__name__)


def create_flow_block(
    input_dimensions: list[int],
    cond_dimensions: list[int] | None,
    flow_steps: int,
    subnet_type: str,
    flow_type: str,
    clamp: float = 2.0,
    **kwargs,
) -> SequenceINN:
    """Build a FrEIA ``SequenceINN`` of ActNorm, random permute, and coupling blocks."""
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        kernel_size = 1 if i % 2 == 1 else 3
        nodes.append(ActNorm)
        nodes.append(PermuteRandom)
        if flow_type == "simple":
            nodes.append(
                AffineCouplingOneSided,
                cond=(0 if cond_dimensions is not None else None),
                cond_shape=cond_dimensions,
                subnet_constructor=build_subnet(
                    subnet_type,
                    hidden_ratio=1.0,
                    kernel_size=kernel_size,
                    **kwargs,
                ),
                clamp=clamp,
                clamp_activation="TANH",
            )
        elif flow_type == "glow":
            nodes.append(
                GLOWCouplingBlock,
                subnet_constructor=build_subnet(
                    subnet_type,
                    hidden_ratio=1.0,
                    kernel_size=kernel_size,
                    **kwargs,
                ),
                clamp=clamp,
                clamp_activation="TANH",
            )
        else:
            raise ValueError(f"Flow type {flow_type} is not supported.")
    return nodes


class FlowCLASModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        backbone_cfg: dict,
        proj_dim: int,
        num_classes: int,
        backbone_ckpt: str | None = None,
        pretrained: bool = True,
        flow_steps: int = 8,
        subnet_type: str = "simple",
        flow_type: str = "simple",
        use_seg: bool = False,
        layers: list[int] = [0, 1, 2, 3],
    ) -> None:
        super().__init__()
        self.flow_type = flow_type

        #* Create the feature extractor
        self.feature_extractor, self.channels, self.scales = build_feature_extractor(backbone, 
                                                                            backbone_cfg=backbone_cfg,
                                                                            ckpt=backbone_ckpt,
                                                                            pretrained=pretrained,
                                                                            layers=layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.flow_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.base_dists = nn.ModuleList()
        self.proj_heads = nn.ModuleList()
        self.seg_heads = nn.ModuleList() if use_seg else None
        for channel, _scale in zip(self.channels, self.scales, strict=True):
            #* Layer normalization
            self.norms.append(nn.LayerNorm((channel,)))
            if self.flow_type == "residual":
                logger.info("Using residual flows. Ignoring subnet type and using Lipschitz CNN.")
                nf_blocks = nn.ModuleList()
                for k in range(flow_steps):
                    kernel_size = 1 if k % 2 == 1 else 3
                    net = nf.nets.LipschitzCNN([channel, channel, channel], [kernel_size, kernel_size])
                    nf_blocks.append(nf.flows.Residual(net))
                    nf_blocks.append(ActNormV2((channel, 1, 1)))
                    nf_blocks.append(nf.flows.mixing.Invertible1x1Conv(channel))
                q0 = MultivariateNormal(channel)
                self.flow_blocks.append(nf.NormalizingFlow(q0, flows=nf_blocks))
            else:
                self.flow_blocks.append(
                    create_flow_block(
                        input_dimensions=(channel, 1, 1),
                        cond_dimensions=None,
                        flow_steps=flow_steps,
                        subnet_type=subnet_type,
                        flow_type=flow_type,
                        pre_norm=False
                    ),
                )

                self.base_dists.append(MultivariateNormal(channel))

            #* Projection head for contrastive loss
            proj_head = nn.Conv2d(channel, proj_dim, kernel_size=1)
            kaiming_init(proj_head, a=0, nonlinearity='relu')
            self.proj_heads.append(proj_head)
            
            #* Segmentation head
            if use_seg:
                seg_head = nn.Sequential(
                    nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channel // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // 2, num_classes, kernel_size=1)
                )
                kaiming_init(seg_head, a=0, nonlinearity='relu')
                self.seg_heads.append(seg_head)

    def forward(self, input_tensor: Tensor) -> dict[str, list[Tensor]]:
        """Run backbone and per-scale flows; return dict keys depend on ``self.training``."""
        # Feature extraction
        self.feature_extractor.eval()
        features = self.feature_extractor(input_tensor)
        if not isinstance(features, list) and not isinstance(features, tuple):
            features = [features]
        if self.norms is not None:
            features = [
                self.norms[i](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for i, feat in enumerate(features)
            ]

        z_list: list[Tensor] = []
        z_sample_list: list[Tensor] = []
        log_jacobian_list: list[Tensor] = []
        log_p_list: list[Tensor] = []
        seg_logit_list: list[Tensor] = []

        for i, feature in enumerate(features):
            if self.flow_type == "residual":
                flow = self.flow_blocks[i]
                z, log_jacobian = flow.inverse_and_log_det(feature.contiguous())
                log_p_z, _ = flow.q0.log_prob(z)
            else:
                z, log_jacobian = self.flow_blocks[i](feature)
                log_p_z, _ = self.base_dists[i].log_prob(z)

            z_list.append(z)
            log_jacobian_list.append(log_jacobian)
            log_p_list.append(log_p_z)

            if self.training:
                z_sample = self.proj_heads[i](z)
                z_sample = F.normalize(z_sample, p=2, dim=1)
                z_sample_list.append(z_sample)

                seg_logits = self.seg_heads[i](z) if self.seg_heads is not None else None
                seg_logit_list.append(seg_logits)

        if not self.training:
            return {"features": features, 
                    "log_p": log_p_list,
                    "z": z_list}

        return {"log_p": log_p_list, 
                "log_jacobians": log_jacobian_list,
                "features": features,
                "z": z_sample_list,
                "seg_logits": seg_logit_list}

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        destination = kwargs.get("destination", None)
        keys = [k for k in state.keys() if "feature_extractor" in k]
        for key in keys:
            state.pop(key)
            if destination is not None and key in destination:
                destination.pop(key)
        return state