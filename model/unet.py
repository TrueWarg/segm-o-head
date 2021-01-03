import torch
from torch import nn
from torchvision import transforms


class BaseUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self._conv1 = self._contraction_block(in_channels, out_channels=64)
        self._conv2 = self._contraction_block(in_channels=64, out_channels=128)
        self._conv3 = self._contraction_block(in_channels=128, out_channels=256)
        self._conv4 = self._contraction_block(in_channels=256, out_channels=512)

        self.maxpool = nn.MaxPool2d((2, 2), stride=2)

        self._upconv4 = self._expansion_block(in_channels=512, middle_channels=1024, out_channels=512)
        self._upconv3 = self._expansion_block(in_channels=1024, middle_channels=512, out_channels=256)
        self._upconv2 = self._expansion_block(in_channels=512, middle_channels=256, out_channels=128)
        self._upconv1 = self._expansion_block(in_channels=256, middle_channels=128, out_channels=64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_result = self._conv1(x)
        conv2_result = self._conv2(self.maxpool(conv1_result))
        conv3_result = self._conv3(self.maxpool(conv2_result))
        conv4_result = self._conv4(self.maxpool(conv3_result))

        upconv4_result = self._upconv4(self.maxpool(conv4_result))

        cropped = transforms.CenterCrop(upconv4_result.shape[2])(conv4_result)
        upconv3_result = self._upconv3(torch.cat([upconv4_result, cropped], 1))

        cropped = transforms.CenterCrop(upconv3_result.shape[2])(conv3_result)
        upconv2_result = self._upconv2(torch.cat([upconv3_result, cropped], 1))

        cropped = transforms.CenterCrop(upconv2_result.shape[2])(conv2_result)
        upconv1_result = self._upconv1(torch.cat([upconv2_result, cropped], 1))

        cropped = transforms.CenterCrop(upconv1_result.shape[2])(conv1_result)
        final_output = self.final_conv(torch.cat([upconv1_result, cropped], 1))

        return final_output

    def _contraction_block(self, in_channels: int, out_channels: int, conv_ksize=(3, 3)) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_ksize),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, conv_ksize),
            nn.ReLU(),
        )

    def _expansion_block(
            self,
            in_channels: int,
            middle_channels: int,
            out_channels: int,
            conv_ksize=(3, 3),
            up_conv_ksize=(2, 2),
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, conv_ksize),
            nn.ReLU(),
            nn.Conv2d(middle_channels, middle_channels, conv_ksize),
            nn.ReLU(),
            nn.ConvTranspose2d(middle_channels, out_channels, up_conv_ksize, stride=2),
        )
