import torch
from torch import nn


class ClassicUnet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self._conv1 = self._contraction_block(in_channels, out_channels=64)
        self._conv2 = self._contraction_block(in_channels=64, out_channels=128)
        self._conv3 = self._contraction_block(in_channels=128, out_channels=256)
        self._conv4 = self._contraction_block(in_channels=256, out_channels=512)

        self._upconv1 = self._expansion_block(in_channels=512, out_channels=1024)
        self._upconv2 = self._expansion_block(in_channels=1024, out_channels=512)
        self._upconv3 = self._expansion_block(in_channels=512, out_channels=256)
        self._upconv4 = self._expansion_block(in_channels=256, out_channels=128)
        self._upconv5 = self._expansion_block(in_channels=128, out_channels=64)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=2)

    def forwart(self, x: torch.Tensor) -> torch.Tensor:
        conv1_result = self._conv1(x)
        conv2_result = self._conv2(conv1_result)
        conv3_result = self._conv3(conv2_result)
        conv4_result = self._conv4(conv3_result)

        upconv1_result = self._upconv1(conv4_result)
        upconv2_result = self._upconv2(torch.cat([upconv1_result, conv4_result], 1))
        upconv3_result = self._upconv3(torch.cat([upconv2_result, conv3_result], 1))
        upconv4_result = self._upconv4(torch.cat([upconv3_result, conv2_result], 1))
        upconv5_result = self._upconv5(torch.cat([upconv4_result, conv1_result], 1))

        final_output = self.final_conv(upconv5_result)

        return final_output

    def _contraction_block(self,
                           in_channels: int,
                           out_channels: int,
                           conv_ksize=(3, 3),
                           pool_ksize=(2, 2),
                           ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_ksize),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, conv_ksize),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, conv_ksize),
            nn.MaxPool2d(pool_ksize, stride=2),
        )

    def _expansion_block(self, in_channels: int,
                         out_channels: int,
                         conv_ksize=(3, 3),
                         up_conv_ksize=(2, 2),
                         ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_ksize),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, conv_ksize),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, conv_ksize),
            nn.ConvTranspose2d(out_channels, out_channels, up_conv_ksize, stride=2),
        )
