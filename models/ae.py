
import torch.nn as nn

# Note: [0, +1] --> nn.Sigmoid()
# Note: [-1, +1] --> nn.Tanh()


class AELinear(nn.Module):
    def __init__(self, input_ftrs=1024, init_ftrs=128, latent_dim=2):
        super(AELinear, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_ftrs, out_features=init_ftrs),
            nn.ReLU(),
            nn.Linear(in_features=init_ftrs, out_features=int(init_ftrs / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(init_ftrs / 2),
                      out_features=int(init_ftrs / 4)),
            nn.ReLU(),
            nn.Linear(in_features=int(init_ftrs / 4), out_features=latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=int(init_ftrs / 4)),
            nn.ReLU(),
            nn.Linear(in_features=int(init_ftrs / 4),
                      out_features=int(init_ftrs / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(init_ftrs / 2), out_features=init_ftrs),
            nn.ReLU(),
            nn.Linear(in_features=init_ftrs, out_features=input_ftrs),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class AEConv3(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 3, 32, 32
        self.encoder = nn.Sequential(
            # (W−F+2P)/S+1
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # N, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # N, 32, 8, 8
            nn.ReLU(),
            nn.Conv2d(32, 64, 8)  # N, 64, 1, 1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 8),  # N, 32, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                               output_padding=1),  # N, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1,
                               output_padding=1),  # N, 3, 32, 32
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
