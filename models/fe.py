from torch import nn


class ShuffleNet_FE(nn.Module):
    def __init__(self, model):
        super(ShuffleNet_FE, self).__init__()

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


class VGG_FE(nn.Module):
    def __init__(self, model):
        super(VGG_FE, self).__init__()
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)

        self.pooling = model.avgpool
        self.flatten = nn.Flatten()
        self.fc = model.classifier[0]

    def forward(self, x):

        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class AELinear_FE(nn.Module):
    def __init__(self, model):
        super(AELinear_FE, self).__init__()

        self.encoder = list(model.encoder)
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        out = self.encoder(x)
        return out
