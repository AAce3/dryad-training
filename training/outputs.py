from typing import Tuple
import torch
import torch.nn as nn
from policy_map import policy_map


class Outputs(nn.Module):
    def __init__(
        self, embedding_size: int, num_policy_heads: int, wdl_dim: int, mlh_dim: int
    ):
        self.policy = AttentionPolicy(embedding_size, num_policy_heads)

        self.wdl = nn.Sequential(
            nn.Linear(embedding_size, wdl_dim),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(wdl_dim * 64, 128),
            nn.Mish(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1),
        )

        self.uncertainty = nn.Sequential(
            nn.Linear(embedding_size, wdl_dim),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(wdl_dim * 64, 128),
            nn.Mish(),
            nn.Linear(128, 1),
            nn.Relu(),
        )

        self.mlh = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(mlh_dim * 64, 128),
            nn.Mish(),
            nn.Linear(128, 1),
        )

        for output_head in [self.wdl, self.uncertainty, self.mlh]:
            for layer in output_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy = self.policy.forward(x)
        wdl = self.wdl.forward(x)
        mlh = self.mlh.forward(x)
        uncertainty = self.uncertainty.forward(x)
        return policy, wdl, mlh, uncertainty


class AttentionPolicy(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int):
        super().__init__()

        self.input_ffn = nn.Linear(embedding_size, embedding_size)
        self.mish = nn.Mish()

        # attention policy
        self.fc_q = nn.Linear(embedding_size, embedding_size)
        self.fc_k = nn.Linear(embedding_size, embedding_size)

        self.promotion_layer = nn.Linear(embedding_size, 4, bias=False)
        self.head_dim = embedding_size // num_heads

        self.register_buffer("attention_map", torch.Tensor(policy_map))

        for layer in [self.fc_k, self.fc_q, self.promotion_layer]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_ffn(x)
        x = self.mish(x)

        batch_size = x.size(0)

        q = self.fc_q(x)
        k = self.fc_k(x)

        q = q.view(batch_size, 64, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, 64, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 64 x 64 move logits for every from_square, to_square pair
        # [Batch, num_heads, 64, 64]
        move_logits = torch.matmul(q, k.transpose(-2, -1))
        move_logits = move_logits.mean(dim=1)

        # retrieve 8x8 slice of every single rank 7 to rank 8 move to calculate pawn promotion logits
        promo_slice = move_logits[:, -16:-8, -8:]  # [Batch, 8, 8]

        pawn_promo_keys = k[:, -8:, :]  # [Batch, 8, embedding_size]
        promo_offsets = self.promotion_layer(pawn_promo_keys)  # [Batch, 8, 4]
        promo_offsets = promo_offsets.transpose(1, 2)  # [Batch, 4, 8]

        knight_promo_logits = promo_slice + promo_offsets[:, 3:4, :]
        queen_promo_logits = knight_promo_logits + promo_offsets[:, 0:1, :]
        rook_promo_logits = knight_promo_logits + promo_offsets[:, 1:2, :]
        bishop_promo_logits = knight_promo_logits + promo_offsets[:, 2:3, :]

        pawn_promo_logits = torch.cat(
            [queen_promo_logits, rook_promo_logits, bishop_promo_logits], dim=2
        )
        move_logits = move_logits.reshape(batch_size, 64 * 64)
        pawn_promo_logits = pawn_promo_logits.reshape(batch_size, 8 * 24)

        # final logits are [batch, 4288], which are mapped to 1858 possible outputs using gather
        final_logits = torch.cat([move_logits, pawn_promo_logits], dim=1)
        assert final_logits.size(1) == 4288
        indices = self.attention_map.unsqueeze(0).expand(batch_size, 1858)
        return torch.gather(final_logits, 1, indices)
