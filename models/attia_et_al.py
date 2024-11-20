import torch
import torch.nn as nn

class AttiaNetwork(nn.Module):
    def __init__(self, num_channels):
        super(AttiaNetwork, self).__init__()
        self.num_channels = num_channels
        self.temporal_net = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.spatial_net = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(16*64*18, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        for i in range(self.num_channels):
            x_channel = x[:, i, :].unsqueeze(1)
            x_channel = self.temporal_net(x_channel)
            x_channel = x_channel.unsqueeze(1)
            if i == 0:
                x_temporal = x_channel
            else:
                x_temporal = torch.cat((x_temporal, x_channel), dim=1)
        x_spatial = self.spatial_net(x_temporal)
        x = self.fc(x_spatial)
        
        return x



