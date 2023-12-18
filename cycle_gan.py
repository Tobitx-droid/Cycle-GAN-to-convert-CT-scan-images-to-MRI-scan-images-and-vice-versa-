"""
    Create the model classes (discriminator and Generator)
"""

import torch.nn as nn
from model.constants import *


##############################################
# Residual block with two convolution layers.
##############################################
class ResidualBlock(nn.Module):
    """
        Residual block class implementation
    """

    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(
                1
            ),  # Reflection padding is used because it gives better image quality at edges.
            nn.Conv2d(
                in_channel, in_channel, 3
            ), 
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
        )

    def forward(self, x):
        """
            Forward pass of the residual block
        """
        return x + self.block(x)


##############################################
# Generator
##############################################
# Generator with 9 residual blocks consists of: -> ()
# c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,R256,R256,R256,
# u128, u64,c7s1-3
##############################################


class GeneratorResNet(nn.Module):
    """
        Generator model class implementation
    """

    def __init__(self, input_shape, num_residual_blocks):
        """
            Initialize the generator model for the cycle consistent GAN
        """
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_channels = 64
        # define a variable 'model' which will continue to update
        # throughout the 3 blocks of Residual -> Downsampling -> Upsampling
        # First c7s1-64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        in_channels = out_channels

        # Down-sampling
        # d128 => d256
        for _ in range(2):
            out_channels *= 2
            model += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Residual blocks
        # R256,R256,R256,R256,R256,R256,R256,R256,R256
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_channels)]

        # Up-sampling
        # u128 => u64
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Output layer
        # c7s1-3
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_channels, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
            Forward pass of the generator model
        """
        return self.model(x)


#######################################################################################################
# #   Discriminator
# # We use 70 × 70 PatchGAN.
# # Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.
# # After the last layer, we apply a convolution to produce a 1-dimensional output.
# # We do not use InstanceNorm for the first C64 layer.
# #  We use leaky ReLUs with a slope of 0.2. The discriminator architecture is:" C64-C128-C256-C512 "
#######################################################################################################
class Discriminator(nn.Module):
    """
        Discriminator model class implementation
    """

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=KERNEL_SIZE,
                    stride=STRIDE,
                    padding=1
                )
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, out_channels=64, normalize=False),
            *discriminator_block(64, out_channels=128),
            *discriminator_block(128, out_channels=256),
            *discriminator_block(256, out_channels=512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(
                # Update this to the correct number of output channels from the last discriminator_block
                in_channels=512,
                out_channels=1,
                kernel_size=KERNEL_SIZE,
                padding=PADDING
            )
        )

    def forward(self, img):
        return self.model(img)
