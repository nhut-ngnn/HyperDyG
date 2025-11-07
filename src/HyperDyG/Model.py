import torch
import torch.nn as nn
from .projection_head import ProjectionHead
from .cross_model import CrossModalEncoders
from .classifier import MLPClassifier
from .DynamicGMU import DynamicGMU
from .HyperGraph import DualHypergraphModule
from .RGCNLayer import RGCNLayer
from .GraphTransformer import GraphTransformerLayer

class HyperDyG(nn.Module):
    def __init__(
        self,
        text_input_dim=768,
        audio_input_dim=768,
        fusion_dim=512,
        projection_dim=512,
        num_heads=4,
        dropout=0.3,
        linear_layer_dims=[512, 256],
        num_classes=4,
        hypergraph_k_text=5,
        hypergraph_k_audio=5,
        hypergraph_threshold=0.5,
        hypergraph_threshold_text=None,
        hypergraph_threshold_audio=None,
        use_rgcn=True,
        use_gtr=True,
        num_relations=3,
    ):
        super().__init__()

        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.encoders = CrossModalEncoders(
            text_input_dim, audio_input_dim, fusion_dim, dropout, num_heads
        )

        self.hypergraph = DualHypergraphModule(
            dim=fusion_dim,
            K_text=hypergraph_k_text,
            K_audio=hypergraph_k_audio,
            normalize=True,
            proj_hidden=None,
            threshold=hypergraph_threshold,
            threshold_text=hypergraph_threshold_text,
            threshold_audio=hypergraph_threshold_audio,
        )

        self.use_rgcn = use_rgcn
        self.use_gtr = use_gtr

        if self.use_rgcn:
            self.rgcn = RGCNLayer(
                in_dim=fusion_dim,
                out_dim=fusion_dim,
                num_rels=num_relations,
                dropout=dropout
            )

        if self.use_gtr:
            self.graph_transformer = GraphTransformerLayer(
                dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout
            )

        self.hyper_expand = nn.Linear(fusion_dim, fusion_dim * 2)

        self.gmu = DynamicGMU(
            text_dim=fusion_dim * 2,
            audio_dim=fusion_dim * 2,
            fusion_dim=fusion_dim
        )

        self.shared_proj = ProjectionHead(
            input_dim=fusion_dim,
            projection_dim=projection_dim
        )

        self.classifier = MLPClassifier(
            input_dim=fusion_dim,
            layer_dims=linear_layer_dims,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, text_feat, audio_feat, return_cls=False, return_all=False, record_stats=True):
        encode_text = self.text_encoder(text_feat)
        encode_audio = self.audio_encoder(audio_feat)

        hyper_text, hyper_audio = self.hypergraph(encode_text, encode_audio, record_stats=record_stats)
        combined = torch.cat([hyper_text, hyper_audio], dim=1)

        if self.use_rgcn:
            adj_list = [
                torch.eye(combined.size(1), device=combined.device).unsqueeze(0).repeat(combined.size(0), 1, 1),
                torch.eye(combined.size(1), device=combined.device).unsqueeze(0).repeat(combined.size(0), 1, 1),
                torch.ones(combined.size(0), combined.size(1), combined.size(1), device=combined.device)
            ]
            combined = self.rgcn(combined, adj_list)

        if self.use_gtr:
            combined = self.graph_transformer(combined)

        hyper_pooled = combined.mean(dim=1)
        hyper_expanded = self.hyper_expand(hyper_pooled)

        text_attn, audio_attn = self.encoders(text_feat, audio_feat)
        text_pooled = text_attn.mean(dim=1)
        audio_pooled = audio_attn.mean(dim=1)
        cross_feat = torch.cat([text_pooled, audio_pooled], dim=-1)

        fusion_vec = self.gmu(hyper_expanded, cross_feat)
        logits = self.classifier(fusion_vec)

        if return_cls:
            return logits

        if return_all:
            return {
                "text_pool": text_pooled,
                "audio_pool": audio_pooled,
                "fusion": fusion_vec,
                "logits": logits,
                "hyper_feat": hyper_pooled
            }

        return logits