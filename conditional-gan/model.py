import torch.nn as nn
import torch
import torch.nn.functional as F

EMBSIZE = 20


class MLPGenerator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPGenerator, self).__init__()
        input_dim += EMBSIZE
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.emb = nn.Embedding(10, EMBSIZE)

    def forward(self, x, y):
        emb = self.emb(y)
        x = torch.cat((x, emb), dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        return torch.tanh(self.fc4(x))


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPDiscriminator, self).__init__()
        input_dim += EMBSIZE
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.emb = nn.Embedding(10, EMBSIZE)

    def forward(self, x, y):
        emb = self.emb(y)
        x = torch.cat((x, emb), dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        return torch.sigmoid(self.fc4(x))
