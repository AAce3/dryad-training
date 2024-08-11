import math
import torch
import torch.nn as nn


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int):
        from model import beta, deepnorm_init

        super().__init__()
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.relative_position_qk = RelativePosition(embedding_dim, 1)
        self.relative_position_v = RelativePosition(embedding_dim, beta(num_layers))

        self.fc_q = nn.Linear(embedding_dim, embedding_dim)
        self.fc_k = nn.Linear(embedding_dim, embedding_dim)

        self.logits_bias = nn.Parameter(torch.Tensor(num_heads, 64, 64))

        self.fc_v = nn.Linear(embedding_dim, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

        for layer in [self.fc_q, self.fc_k, self.fc_v, self.fc_out]:
            if layer in [self.fc_q, self.fc_k]:
                nn.init.xavier_normal_(layer.weight)
            else:
                deepnorm_init(layer.weight, num_layers)
            nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.logits_bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        # standard attention dimensions: [batch, num_heads, 64, head_dim]
        q_attn = q.view(batch_size, 64, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        k_attn = k.view(batch_size, 64, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        # compute a grid of attention pairs: [batch, num_heads, 64, 64]
        attn_logits = torch.matmul(q_attn, k_attn.transpose(-2, -1))

        # [64, 64, head_dim]
        relative_embeddings_qk = self.relative_position_qk()

        # Compute relative Q logits

        # result: [batch, num_heads, 64, 64]
        relative_logits_q = self.compute_relative_qk_logits(
            q_attn, relative_embeddings_qk
        )

        # Compute relative K logits
        # note that we have to transpose, then transpose again
        # result: [batch, num_heads, 64, 64]
        relative_logits_k = self.compute_relative_qk_logits(
            k_attn, relative_embeddings_qk.transpose(0, 1)
        ).transpose(2, 3)

        final_logits = (
            attn_logits + relative_logits_q + relative_logits_k + self.logits_bias
        )

        final_logits = final_logits / math.sqrt(self.head_dim)

        attention_weights = torch.softmax(final_logits, dim=-1)

        # [batch_size, num_heads, 64, head_dim]
        v_attn = v.view(batch_size, 64, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        # [batch_size, num_heads, 64, head_dim]
        z_attn = torch.matmul(attention_weights, v_attn)

        # compute relative z logits

        # [batch_size, num_heads, 64, head_dim]
        z_relative = self.compute_relative_z(
            attention_weights, self.relative_position_v()
        )

        z_final = z_relative + z_attn

        z_final = z_final.permute(0, 2, 1, 3).reshape(
            batch_size, 64, self.embedding_dim
        )
        output = self.fc_out(z_final)

        return output

    def compute_relative_qk_logits(
        self, vector: torch.Tensor, relative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # relative vector: [batch, num_heads, 64, 1, head_dim]
        relative_vector = vector.unsqueeze(3)
        # relative embeddings: [num_heads, 64, head_dim, 64]
        relative_embeddings = relative_embeddings.view(
            64, 64, self.num_heads, self.head_dim
        )
        relative_embeddings = relative_embeddings.permute(2, 0, 3, 1)

        # matmul output: [batch, num_heads, 64, 1, 64]
        relative_logits = torch.matmul(relative_vector, relative_embeddings)

        # final result: [batch, num_heads, 64, 64]
        relative_logits = relative_logits.squeeze(3)
        return relative_logits

    def compute_relative_z(
        self, attn_scores: torch.Tensor, relative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # relative_scores: [batch, num_heads, 64, 1, 64]
        relative_scores = attn_scores.unsqueeze(3)

        # relative embeddings: [num_heads, 64, 64, head_dim]
        relative_embeddings = relative_embeddings.view(
            64, 64, self.num_heads, self.head_dim
        )
        relative_embeddings = relative_embeddings.permute(2, 0, 1, 3)

        # result: [batch, num_heads, 64, 1, head_dim]
        relative_z = torch.matmul(relative_scores, relative_embeddings)
        relative_z = relative_z.squeeze(3)

        return relative_z


# Embeddings are stored as a 15x15 grid, representing all possible pair-distances between all possible squares
# In order to use this for efficient matrix computation, we need to expand it out into a [64, 64, embed_dim]
# tensor, representing the weights between each pair of squares. This does involve duplicating.
# for example, the weight between a1 and h1 should be the same as the weight between a8 and h8, due to the
# fact that they are the same distance as each other.
class RelativePosition(nn.Module):
    def __init__(self, head_dim: int, xavier_gain: float):
        super().__init__()
        self.embedding_dim = head_dim
        self.embeddings = nn.Parameter(torch.Tensor(15, 15, head_dim))
        nn.init.xavier_uniform_(self.embeddings, gain=xavier_gain)

        range_vec = torch.arange(8)
        distance_matrix = (range_vec[None, :] - range_vec[:, None]).long() + 7
        self.register_buffer("distance_matrix", distance_matrix)

        row_distances = distance_matrix.repeat_interleave(8, dim=0).repeat_interleave(
            8, dim=1
        )
        col_distances = distance_matrix.repeat(8, 8)

        self.register_buffer("row_distances", row_distances)
        self.register_buffer("col_distances", col_distances)

    def forward(self) -> torch.Tensor:
        return self.embeddings[self.row_distances, self.col_distances]
