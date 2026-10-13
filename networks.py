import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Shared body with one softmax head per independent action choice."""

    def __init__(self, input_size, head_sizes, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.heads = nn.ModuleList(nn.Linear(hidden, size) for size in head_sizes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return [torch.softmax(head(x), dim=-1) for head in self.heads]


class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze()
