import torch
from torch import nn


class Conv2DCompressor(nn.Module):
    """
    Compresse les feature maps en utilisant un module CNN à 2 couches.
    Implémente le sous-échantillonnage spatial 16x (H/4, W/4) et
    l'augmentation des canaux de 256 à 1024, comme décrit dans le papier.

    (In: B, 256, 64, 64) -> (Out: B, 1024, 16, 16)
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 1024):
        super(Conv2DCompressor, self).__init__()

        # Le papier n'est pas explicite sur le nombre de canaux intermédiaires.
        # Une progression logique est 256 -> 512 -> 1024.
        intermediate_channels = (
            in_channels + out_channels
        ) // 2  # (256+1024)//2 = 640 ou 512
        intermediate_channels = 512  # Forçons une valeur standard

        self.compressor = nn.Sequential(
            # --- Couche 1 ---
            # Paper: kernel=3, stride=2, padding=1
            # Input: (B, 256, 64, 64)
            # Output: (B, 512, 32, 32) (Downsample 4x)
            nn.Conv2d(
                in_channels, intermediate_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.GELU(),  # Activation (GELU est courant avec les Transformers)
            # --- Couche 2 ---
            # Paper: kernel=3, stride=2, padding=1
            # Input: (B, 512, 32, 32)
            # Output: (B, 1024, 16, 16) (Downsample 16x total)
            nn.Conv2d(
                intermediate_channels, out_channels, kernel_size=3, stride=2, padding=1
            ),
            # Pas d'activation finale, la sortie est l'embedding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresser la feature map

        Args:
            x: Tenseur d'entrée (B, C_in, H, W), ex: (B, 256, 64, 64)

        Returns:
            Tenseur compressé (B, C_out, H_out, W_out), ex: (B, 1024, 16, 16)
        """
        return self.compressor(x)
