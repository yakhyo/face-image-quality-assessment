import torch
import torch.nn as nn


class ConvBNPReLU(nn.Module):
    """Conv2d + BatchNorm2d + PReLU (optional)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class InvertedResidual(nn.Module):
    """Inverted residual: 1x1 expand -> depthwise -> 1x1 project (+ optional residual)."""

    def __init__(
        self,
        in_channels: int,
        expand_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 2,
        padding: int | tuple[int, int] = 1,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.conv = ConvBNPReLU(in_channels, expand_channels, kernel_size=1)
        self.conv_dw = ConvBNPReLU(
            expand_channels, expand_channels, kernel_size, stride, padding, groups=expand_channels
        )
        self.project = ConvBNPReLU(expand_channels, out_channels, kernel_size=1, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.conv_dw(out)
        out = self.project(out)
        if self.residual:
            out = out + x
        return out


class ResidualStack(nn.Module):
    """Stack of N InvertedResidual blocks with residual=True."""

    def __init__(self, channels: int, num_blocks: int, expand_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            *[
                InvertedResidual(channels, expand_channels, channels, stride=1, residual=True)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# IResNet (eDifFIQA S/M/L)


class IBasicBlock(nn.Module):
    """Pre-activation 3x3-3x3 residual block used by IResNet."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        return out + identity


class IResNet(nn.Module):
    """Improved ResNet (InsightFace flavor). Input must be 112x112."""

    fc_scale = 7 * 7

    def __init__(self, layers: list[int], num_features: int = 512) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(0.0, inplace=True)

        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        x = self.fc(x)
        x = self.features(x)
        return x


class IResNet18(IResNet):
    """IResNet-18 backbone (eDifFIQA-S)."""

    def __init__(self) -> None:
        super().__init__(layers=[2, 2, 2, 2])


class IResNet50(IResNet):
    """IResNet-50 backbone (eDifFIQA-M)."""

    def __init__(self) -> None:
        super().__init__(layers=[3, 4, 14, 3])


class IResNet100(IResNet):
    """IResNet-100 backbone (eDifFIQA-L)."""

    def __init__(self) -> None:
        super().__init__(layers=[3, 13, 30, 3])


# MobileFaceNet (eDifFIQA T)
class MobileFaceNet(nn.Module):
    """MobileFaceNet backbone (eDifFIQA-T). Input must be 112x112."""

    def __init__(self, embedding_size: int = 512) -> None:
        super().__init__()
        self.conv1 = ConvBNPReLU(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_dw = ConvBNPReLU(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.conv_23 = InvertedResidual(64, expand_channels=128, out_channels=64)
        self.conv_3 = ResidualStack(64, num_blocks=4, expand_channels=128)
        self.conv_34 = InvertedResidual(64, expand_channels=256, out_channels=128)
        self.conv_4 = ResidualStack(128, num_blocks=6, expand_channels=256)
        self.conv_45 = InvertedResidual(128, expand_channels=512, out_channels=128)
        self.conv_5 = ResidualStack(128, num_blocks=2, expand_channels=256)
        self.conv_6_sep = ConvBNPReLU(128, 512)
        self.conv_6_dw = ConvBNPReLU(512, 512, kernel_size=7, groups=512, activation=False)
        self.conv_6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2_dw(x)

        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)

        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)

        x = self.linear(x)
        x = self.bn(x)

        return nn.functional.normalize(x, p=2, dim=1)
