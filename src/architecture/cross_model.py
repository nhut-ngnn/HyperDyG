import torch
import torch.nn as nn

class CrossModalEncoders(nn.Module):
    def __init__(self, text_input_dim, audio_input_dim, fusion_dim, dropout, num_heads):
        super().__init__()
        self.cross_attention_text = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_audio = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, text_feat, audio_feat):
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)


        text_attn, _ = self.cross_attention_text(text_feat, audio_feat, audio_feat)
        audio_attn, _ = self.cross_attention_audio(audio_feat, text_feat, text_feat)

        return text_attn, audio_attn