from typing import Tuple
import torch
import torch.nn as nn
import attention
from outputs import Outputs


class Model(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_expansion_factor: int,
        num_layers: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            InputEncoder(embedding_dim, ffn_expansion_factor, num_heads)
        )
        for _ in range(num_layers):
            self.layers.append(
                AttentionBlock(
                    embedding_dim, num_heads, ffn_expansion_factor, num_layers
                )
            )
        self.layers.append(Outputs(embedding_dim, num_heads, 64, 16))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.layers.forward(x)


class InputEncoder(nn.Module):
    def __init__(self, embedding_dim: int, ffn_expansion_factor: int, num_layers: int):
        super().__init__()
        self.embed_dim = embedding_dim
        self.preprocess = nn.Linear(64 * 12, 64 * embedding_dim)
        self.fc_1 = nn.Linear(64 + embedding_dim, embedding_dim)
        self.mish = nn.Mish()
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.add_gate = nn.Parameter(torch.Tensor(64, embedding_dim))
        self.mul_gate = nn.Parameter(torch.Tensor(64, embedding_dim))

        self.ffn = FFN(embedding_dim, ffn_expansion_factor, num_layers)
        self.alpha = alpha(num_layers)

        for layer in [self.preprocess, self.fc_1]:
            nn.init.xavier_normal_(layer.weight)

    # x: [batch, 112, 8, 8]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, 112, 64).transpose(1, 2)  # [ batch, 64, 112 ]
        # the first 12 channels are the first position

        embedding_x = x[:, :, :12].reshape(batch_size, 64 * 12)

        encoding = self.preprocess(embedding_x).reshape(batch_size, 64, self.embed_dim)

        x = torch.cat(x, encoding, dim=2)
        x = self.fc_1(
            x
        )  # projects the 112 + embedding_dim vectors back down to embedding_dim
        x = self.mish(x)
        x = self.layernorm(x)
        x = x * self.mul_gate
        x = x + self.add_gate
        ffn = self.ffn(x)
        x = x + self.alpha * ffn
        x = self.layernorm(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_expansion_factor: int,
        num_layers: int,
    ):
        super().__init__()
        self.attention = attention.RelativeMultiHeadAttention(
            embedding_dim, num_heads, num_layers
        )
        self.ffn = FFN(embedding_dim, ffn_expansion_factor, num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.alpha = alpha(num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_x = self.attention(x)
        x = x + self.alpha * attn_x
        x = self.layer_norm(x)
        ffn_x = self.ffn(x)
        x = x + self.alpha * ffn_x
        x = self.layer_norm(x)
        return x


class FFN(nn.Module):
    def __init__(self, embedding_dim: int, expansion_factor: int, num_layers: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * expansion_factor),
            nn.Mish(),
            nn.Linear(embedding_dim * expansion_factor, embedding_dim),
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                deepnorm_init(layer.weight, num_layers)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(x)


# Functions for deepnet initalization https://arxiv.org/pdf/2203.00555
def beta(num_layers: int) -> float:
    return (8.0 * num_layers) ** (-1 / 4)


def alpha(num_layers: int) -> float:
    return (2.0 * num_layers) ** (1 / 4)


def deepnorm_init(tensor: torch.Tensor, num_layers: int):
    nn.init.xavier_normal_(tensor, gain=beta(num_layers))
