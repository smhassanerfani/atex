from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()

        self.conv1 = list(model.conv1)
        self.conv1 = nn.Sequential(*self.conv1)

        self.maxpool = model.maxpool

        self.stage2 = list(model.stage2)
        self.stage2 = nn.Sequential(*self.stage2)

        self.stage3 = list(model.stage3)
        self.stage3 = nn.Sequential(*self.stage3)

        self.stage4 = list(model.stage4)
        self.stage4 = nn.Sequential(*self.stage4)

        self.conv5 = list(model.conv5)
        self.conv5 = nn.Sequential(*self.conv5)

        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.conv5(out)
        out = self.flatten(out)
        return out
