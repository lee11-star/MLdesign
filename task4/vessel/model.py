import torch
from torch import nn
from torch.nn import functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=4, width=16):
        super().__init__()
        self.enc1 = double_conv(in_channels, width)
        self.enc2 = double_conv(width, width * 2)
        self.enc3 = double_conv(width * 2, width * 4)
        self.bottom = double_conv(width * 4, width * 8)
        self.dec3 = double_conv(width * 12, width * 4)
        self.dec2 = double_conv(width * 6, width * 2)
        self.dec1 = double_conv(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottom(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1], dim=1))
        return self.head(d1)


@torch.inference_mode()
def predict_probability(model, features, device, tta=True):
    model.eval()
    x = torch.from_numpy(features[None]).to(device)
    height, width = x.shape[-2:]
    x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8), mode='reflect')
    if tta == 'd4':
        transforms = [(turns, mirror) for turns in range(4) for mirror in (False, True)]
    elif tta:
        transforms = [(0, False), (0, True), (2, True), (2, False)]
    else:
        transforms = [(0, False)]
    result = torch.zeros((1, 1, *x.shape[-2:]), device=device)
    for turns, mirror in transforms:
        augmented = torch.rot90(x, turns, dims=(-2, -1))
        if mirror:
            augmented = torch.flip(augmented, (-1,))
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            probability = model(augmented).sigmoid()
        if mirror:
            probability = torch.flip(probability, (-1,))
        result += torch.rot90(probability, -turns, dims=(-2, -1))
    return (result[0, 0, :height, :width] / len(transforms)).float().cpu().numpy()
