import torch
from torch import nn


class BaseUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = self._contraction_block(in_channels=in_channels, out_channels=32, kernel_size=(7, 7), padding=3)
        self.conv2 = self._contraction_block(in_channels=32, out_channels=64)
        self.conv3 = self._contraction_block(in_channels=64, out_channels=128)

        self.upconv3 = self._expansion_block(in_channels=128, out_channels=64)
        self.upconv2 = self._expansion_block(in_channels=128, out_channels=32)
        self.upconv1 = self._expansion_block(in_channels=64, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def _contraction_block(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=(3, 3),
            padding=1,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def _expansion_block(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=(3, 3),
            padding=1,
    ) -> nn.Module:
        return nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
            )
        )
