import torch
import torch.nn as nn
import torch.nn.functional as F


class UserProfiler(nn.Module):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int = 128,
        user_embedding_dim: int = 64,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.num_items = num_items
        self.item_embedding_dim = item_embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )

        self.rating_proj = nn.Sequential(
            nn.Linear(1, item_embedding_dim),
            nn.SiLU(),
        )

        self.query = nn.Parameter(torch.randn(1, 1, item_embedding_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=item_embedding_dim,
            num_heads=4,
            batch_first=True,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(item_embedding_dim, user_embedding_dim),
            nn.SiLU(),
            nn.Linear(user_embedding_dim, user_embedding_dim),
        )

        self.item_id_to_idx = {}

    def build_item_index(self, all_item_ids: set):
        self.item_id_to_idx = {item_id: idx + 1 for idx, item_id in enumerate(sorted(all_item_ids))}
        actual_num = len(self.item_id_to_idx) + 1
        if actual_num > self.item_embedding.num_embeddings:
            self.item_embedding = nn.Embedding(
                actual_num, self.item_embedding_dim, padding_idx=0
            )
        self.num_items = len(self.item_id_to_idx)

    def forward(
        self,
        item_indices: torch.LongTensor,
        ratings: torch.FloatTensor,
        mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        B, L = item_indices.shape

        item_emb = self.item_embedding(item_indices)

        rating_emb = self.rating_proj(ratings.unsqueeze(-1))

        seq_emb = item_emb + rating_emb

        if mask is None:
            mask = (item_indices != 0)

        query = self.query.expand(B, -1, -1)

        key_padding_mask = ~mask

        attn_out, _ = self.attention(
            query=query,
            key=seq_emb,
            value=seq_emb,
            key_padding_mask=key_padding_mask,
        )

        user_emb = attn_out.squeeze(1)
        user_emb = self.output_proj(user_emb)

        return user_emb
