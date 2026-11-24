"""Self-Supervised and Downstream Raman neural network architectures in PyTorch."""

from .resnet_barlow_twins import (
    BarlowTwins1D,
    BarlowTwinsLoss,
    BasicBlock1D,
    ResNet1DBackbone,
    ResNet1DClassifier,
)
from .vit_smae import (
    FeedForward,
    MultiHeadSelfAttention,
    PatchEmbed1D,
    SpectralMAE,
    SpectralMAEClassifier,
    TransformerBlock,
)

__all__ = [
    "PatchEmbed1D",
    "MultiHeadSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "SpectralMAE",
    "SpectralMAEClassifier",
    "BasicBlock1D",
    "ResNet1DBackbone",
    "BarlowTwinsLoss",
    "BarlowTwins1D",
    "ResNet1DClassifier",
]
