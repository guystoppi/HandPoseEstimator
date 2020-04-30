import torch
from torch import nn

def ceil_floor(x):
  return torch.floor(x), torch.ceil(x)

def pad_to_same_hw(input1, input2):
  if input1.shape[-2] < input2.shape[-2]: # height
    diff = ceil_floor(input2.shape[-2] - input1.shape[-2])

    input1 = nn.functional.pad(input1, (diff[0], diff[1], 0, 0))
  elif input1.shape[-2] > input2.shape[-2]:
    diff = ceil_floor(input1.shape[-2] - input2.shape[-2])

    input2 = nn.functional.pad(input2, (diff[0], diff[1], 0, 0))

  if input1.shape[-1] < input2.shape[-1]: # width
    diff = ceil_floor(input2.shape[-1] - input1.shape[-1])

    input1 = nn.functional.pad(input1, (diff[0], diff[1]))
  elif input1.shape[-1] > input2.shape[-1]:
    diff = ceil_floor(input1.shape[-1] - input2.shape[-1])

    input2 = nn.functional.pad(input2, (diff[0], diff[1]))

  return input1, input2

class DoubleConv(nn.Module):
  
  def __init__(self, in_chan, out_chan):
    super(DoubleConv, self).__init__()

    self.seq = nn.Sequential(
        nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(out_chan),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(out_chan),
        nn.ReLU(inplace=True)
    )

  def forward(self, input):
    return self.seq(input)

class UpConv(nn.Module):
  def __init__(self, in_chan, out_chan):
    super(UpConv, self).__init__()

    self.seq = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        DoubleConv(in_chan, out_chan)
    )

  def forward(self, input1, input2):

    input1 = self.seq(input1)

    input1, input2 = pad_to_same_hw(input1, input2)

    return torch.cat([input1, input2], dim=1)

class DownConv(nn.Module):
  def __init__(self, in_chan, out_chan):
    super(DownConv, self).__init__()

    self.conv = nn.Sequential(      
        nn.MaxPool2d(2),
        DoubleConv(in_chan, out_chan)
    )

  def forward(self, input):
    return self.conv(input)

class Model(nn.Module):

  def __init__(self, in_chan, out_classes):
    super(Model, self).__init__()

    self.in_layer = DoubleConv(in_chan, 32)
    self.down2 = DownConv(32, 64)
    self.down3 = DownConv(64, 128)
    self.down4 = DownConv(128, 256)

    self.up1 = UpConv(256, 128)
    self.up2 = UpConv(256, 64)
    self.up3 = UpConv(128, 32)

    self.conv = DoubleConv(64, 32)
    self.final_layer = nn.Linear(32, out_classes)


  def forward(self, input):
    d1 = self.in_layer(input) # B x 32 x H x W
    d2 = self.down2(d1) # B x 64 x H/2 x W/2
    d3 = self.down3(d2) # B x 128 x H/4 x W/4
    d4 = self.down4(d3) # B x 256 x H/8 x W/8

    u1 = self.up1(d4, d3) # B x 256 x H/4 x W/4
    u2 = self.up2(u1, d2) # B x 128 x H/2 x W/2
    u3 = self.up3(u2, d1) # B x 64 x H x W

    out = self.conv(u3).permute(0, 2, 3, 1) # channel to back
    out1 = self.final_layer(out).permute(0, 3, 1, 2)

    return out1
