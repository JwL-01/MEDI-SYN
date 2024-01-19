import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, num_features):
        super(FiLM, self).__init__()
        # Define layers to generate the FiLM parameters: gamma (scale) and beta (shift)
        self.gamma_layer = nn.Linear(1, num_features)
        self.beta_layer = nn.Linear(1, num_features)
        
    def forward(self, feature_maps, noise_level):
        # The noise_level tensor is expected to be of shape (batch_size, 1)
        # Expand noise_level to match the batch size of feature_maps
        noise_level = noise_level.view(-1, 1).expand(feature_maps.size(0), 1)
        
        # Generate the FiLM parameters
        gamma = self.gamma_layer(noise_level)
        beta = self.beta_layer(noise_level)
        
        # The gamma and beta parameters are expected to be of shape (batch_size, num_features)
        # We add dimensions to gamma and beta to match the dimensions of feature_maps
        gamma = gamma.view(-1, gamma.size(1), 1, 1)
        beta = beta.view(-1, beta.size(1), 1, 1)
        
        # Apply the FiLM transformation: feature_maps * gamma + beta
        # We use broadcasting to apply the same scale and shift to all spatial positions
        modulated_feature_maps = feature_maps * gamma + beta
        
        return modulated_feature_maps

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
    # Note the change in the in_channels parameter to account for the skip connections
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels + skip_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)
        x = self.se(x)
        return x

class StepwiseReverseDiffusionNet(nn.Module):
    def __init__(self):
        super(StepwiseReverseDiffusionNet, self).__init__()

        # Encoder blocks
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # FiLM layer after the encoder and before the decoder
        self.film = FiLM(512)  # Assuming the encoder's last block has 512 channels

        # Decoder blocks
        self.dec1 = DecoderBlock(512, 256)  # Adjusted for FiLM output
        self.dec2 = DecoderBlock(256, 128, skip_channels=256)
        self.dec3 = DecoderBlock(128, 64, skip_channels=128)
        self.dec4 = DecoderBlock(64, 1, skip_channels=64)

    def forward(self, x, noise_level):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Apply FiLM modulation before starting the decoding process
        film_out = self.film(enc4, noise_level)

        # Decoder with skip connections
        dec1 = self.dec1(film_out)
        dec1 = torch.cat((dec1, enc3), 1)  # Skip connection from enc3
        dec2 = self.dec2(dec1)
        dec2 = torch.cat((dec2, enc2), 1)  # Skip connection from enc2
        dec3 = self.dec3(dec2)
        dec3 = torch.cat((dec3, enc1), 1)  # Skip connection from enc1
        dec4 = self.dec4(dec3)
        
        # Final Activation
        out = torch.tanh(dec4)
        return out