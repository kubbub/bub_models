import torch
from torch import nn
from torchinfo import summary


class TransformerEncoderBlock(nn.Module):
    """Single encoder block for Vision Transformer"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # multi head self attention block
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # multilayer perceptron block
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=dim_feedforward),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(in_features=dim_feedforward, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        """single encoder block"""
        # layer norm + multi head self attention block
        norm_x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=norm_x, key=norm_x, value=norm_x, need_weights=False
        )
        x = attn_output + x

        # multi layer perceptron
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module):
    """
    AN IMAGE IS WORTH 16x16 WORDS
    Transformer for image recognition
        **paper replicate**
    """

    def __init__(
        self,
        image_size: int = 224,
        embedding_dim: int = 768,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 3,
        dropout: int = 0.1,
        num_heads: int = 12,
        dim_feedforward: int = 3072,
        num_layers: int = 12,
    ):
        super().__init__()

        # INPUT EMBEDDING
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.class_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim), requires_grad=True
        )
        num_patches = (image_size * image_size) // patch_size**2
        self.embedding_pos = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_dim), requires_grad=True
        )
        self.embedding_dropout = nn.Dropout(p=dropout)

        ## ENCODER
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def _PatchEmbedding(self, x):
        """create patched and flatten represenation of an input image"""
        return (self.flatten(self.patcher(x))).permute(0, 2, 1)

    def _EncoderInput(self, x):
        """create embedding input to encoder layer"""
        batch_size = x.shape[0]
        patched_embedding = self._PatchEmbedding(x)
        ## add class token parameter
        x = torch.cat(
            (self.class_token.expand(batch_size, -1, -1), patched_embedding), dim=1
        )
        ## add embeddingh position
        x = x + self.embedding_pos
        ## include droput on positional and patch embedding
        x = self.embedding_dropout(x)
        return x

    def forward(self, x):
        x = self._EncoderInput(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]  # Get class token
        return self.classifier(x)


if __name__ == "__main__":
    model = ViT()
    summary(
        model=model,
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )
