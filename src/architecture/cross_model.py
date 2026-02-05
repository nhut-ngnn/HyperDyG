import torch
import torch.nn as nn


class CrossModalEncoders(nn.Module):
    def __init__(
        self,
        text_input_dim,
        audio_input_dim,
        fusion_dim,
        dropout,
        num_heads,
        num_blocks=1,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1.")
        self.num_blocks = num_blocks
        self.cross_attention_text = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_blocks)
            ]
        )
        self.cross_attention_audio = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, text_feat, audio_feat):
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)

        for text_attn_layer, audio_attn_layer in zip(
            self.cross_attention_text, self.cross_attention_audio
        ):
            text_out, _ = text_attn_layer(text_feat, audio_feat, audio_feat)
            audio_out, _ = audio_attn_layer(audio_feat, text_feat, text_feat)
            text_feat, audio_feat = text_out, audio_out

        return text_feat, audio_feat
