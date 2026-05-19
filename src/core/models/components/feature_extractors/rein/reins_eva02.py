

import logging
from functools import partial

import torch
import torch.nn.functional as F
from mmengine.runner.checkpoint import _load_checkpoint

from .reins import LoRAReins
from .eva_02 import EVA2

logger = logging.getLogger(__name__)


class ReinsEVA2(EVA2):
    def __init__(self, ckpt: str) -> None:
        super().__init__(
            pretrained="../misc/weights/eva02_L_pt_m38m_p14.pt"
        )

        reins_state_dict = _load_checkpoint(ckpt, logger=logger, map_location='cpu')
        if "state_dict" in reins_state_dict:
            reins_state_dict = reins_state_dict["state_dict"]
        reins_state_dict = {k.replace("backbone.reins.", ""): v for k, v in reins_state_dict.items() if k.startswith("backbone.reins")}
        self.reins = LoRAReins(
            embed_dims=1024,
            link_token_to_query=True,
            lora_dim=16,
            num_layers=24,
            patch_size=16,
            token_length=100
        )
        self.reins.load_state_dict(reins_state_dict, strict=True)
        logger.info("Loaded pretrained weights for Reins.")
        print("Loaded pretrained weights for Reins.")

    def forward(self, x, masks=None):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features = []
        for i, blk in enumerate(self.blocks):
            blk.attn.rope.img_size = (Hp, Wp)
            x = blk(x, rel_pos_bias)
            x = self.reins.forward(
                x,
                i,
                batch_first=True,
                has_cls_token=True,
            )
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
        features[0] = F.interpolate(
            features[0], scale_factor=4, mode="bilinear", align_corners=False
        )
        features[1] = F.interpolate(
            features[1], scale_factor=2, mode="bilinear", align_corners=False
        )
        features[3] = F.interpolate(
            features[3], scale_factor=0.5, mode="bilinear", align_corners=False
        )
        reins_out = self.reins.return_auto(features)
        return reins_out[0][-1] #* return the last layer output