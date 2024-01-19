import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
# Label Embedding Module
class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(LabelEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, label):
        return self.embedding(label)

# Squeeze-and-Excitation Layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)
        x = self.se(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.se = SELayer(out_channels)  # SELayer as defined previously

    def forward(self, x, skip=None):
        if skip is not None:
            skip = F.interpolate(skip, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            x = torch.cat((x, skip), 1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)
        x = self.se(x)
        return x


# Main Model
class StepwiseReverseDiffusionNet2(nn.Module):
    def __init__(self, num_classes, embedding_dim=1):
        super(StepwiseReverseDiffusionNet2, self).__init__()
        self.label_embedding = LabelEmbedding(num_classes, embedding_dim)

        # Encoder blocks
        self.enc1 = EncoderBlock(3 + embedding_dim, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Decoder blocks
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, 3)


        # Final adjustment layers for image generation
        self.to_image = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, label):
        label_embedding = self.label_embedding(label)
        label_embedding = label_embedding.unsqueeze(-1).unsqueeze(-1)
        label_embedding = label_embedding.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, label_embedding], dim=1)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder - without skip connections
        dec1 = self.dec1(enc4)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        x = self.to_image(dec4)
        return x
