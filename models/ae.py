import torch
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
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AEConv4(nn.Module):
    def __init__(self, image_channels=3, init_channels=16, latent_dim=64):
        super(AEConv4, self).__init__()

        # N, 3, 32, 32
        self.encoder = nn.Sequential(
            # (W−F+2P)/S+1: (32-3+2)/2+1 ==> N, 16, 16, 16
            nn.Conv2d(in_channels=image_channels, out_channels=init_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (W−F+2P)/S+1: (16-3+2)/2+1 ==> N, 32, 8, 8
            nn.Conv2d(in_channels=init_channels, out_channels=init_channels *
                      2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (W−F+2P)/S+1: (8-3+2)/2+1 ==> N, 64, 4, 4
            nn.Conv2d(in_channels=init_channels * 2, out_channels=init_channels * \
                      4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (W−F+2P)/S+1: (4-4)/1+1 ==> N, 128, 1, 1
            nn.Conv2d(in_channels=init_channels * 4, out_channels=init_channels * \
                      8, kernel_size=4)
        )

        self.decoder = nn.Sequential(
            # H_out=(H_in−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
            # W_out=(W_in−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1
            nn.ConvTranspose2d(
                in_channels=128, out_channels=init_channels * 4, kernel_size=4),  # N, 64, 4, 4
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=init_channels * 4,
                               out_channels=init_channels * 2, kernel_size=3,
                               stride=2, padding=1, output_padding=1),  # N, 32, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=init_channels * 2,
                               out_channels=init_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),  # N, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=init_channels,
                               out_channels=image_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),  # N, 3, 32, 32
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VAEConv3(nn.Module):
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
            nn.Sigmoid()
            # nn.Tanh()
        )

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, 32)
        self.fc_log_var = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 64)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling

        return sample

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.reshape(encoded.shape[0], -1)

        hidden = self.fc1(encoded)
        # get `mu` and `log_var`c
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)

        decoded = self.decoder(z)

        return decoded, mu, log_var
