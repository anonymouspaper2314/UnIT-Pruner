import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import rtp

class PyTorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(288, 12)

        self.threshold = nn.Threshold(0, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.threshold(x)
        x = self.fc1(x)

        return x
    
    def pruned_count(self):
        pruned = 0
        total = 0

        pruned += torch.sum(self.conv1.weight == 0)
        total += torch.numel(self.conv1.weight)
        pruned += torch.sum(self.conv2.weight == 0)
        total += torch.numel(self.conv2.weight)
        pruned += torch.sum(self.fc1.weight == 0)
        total += torch.numel(self.fc1.weight)

        return pruned, total
        
class RTPNoPruneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = rtp.NPConv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = rtp.NPConv2d(6, 16, 5)
        self.fc1 = rtp.NPLinear(288, 12)

        self.threshold = nn.Threshold(0, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.threshold(x)
        x = self.fc1(x)

        return x
    
    def pruned_count(self):
        pruned = 0
        total = 0

        pruned += torch.sum(self.conv1.weight == 0)
        total += torch.numel(self.conv1.weight)
        pruned += torch.sum(self.conv2.weight == 0)
        total += torch.numel(self.conv2.weight)
        pruned += torch.sum(self.fc1.weight == 0)
        total += torch.numel(self.fc1.weight)
        pruned += torch.sum(self.fc2.weight == 0)
        total += torch.numel(self.fc2.weight)
        pruned += torch.sum(self.fc3.weight == 0)
        total += torch.numel(self.fc3.weight)

        return pruned, total
        
class RTPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = rtp.NPConv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = rtp.NPConv2d(6, 16, 5)
        self.fc1 = rtp.Linear(288, 12)

    def forward(self, x, threshold):
        self.threshold = nn.Threshold(0, 0)
        # self.threshold = nn.Threshold(threshold, 0)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.threshold(x)
        x = self.fc1(x, threshold)
        
        return x

class RTPDebugNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = rtp.NPConv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = rtp.NPConv2d(6, 16, 5)
        self.fc1 = rtp.DebugLinear(288, 12)

    def forward(self, x, threshold, stats):
        self.threshold = nn.Threshold(0, 0)
        # self.threshold = nn.Threshold(threshold, 0)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.threshold(x)
        x = self.fc1(x, threshold, stats)
        
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(288, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)

        return x
        
    def prune(self):
        prune.global_unstructured(
            (
                (self.conv1, 'weight'),
                (self.conv2, 'weight'),
                (self.fc1, 'weight')
            ),
            pruning_method=prune.L1Unstructured,
            amount=0.05,
        )
    
    def pruned_count(self):
        pruned = 0
        total = 0

        pruned += torch.sum(self.conv1.weight == 0)
        total += torch.numel(self.conv1.weight)
        pruned += torch.sum(self.conv2.weight == 0)
        total += torch.numel(self.conv2.weight)
        pruned += torch.sum(self.fc1.weight == 0)
        total += torch.numel(self.fc1.weight)

        return pruned, total
    
    def save_prune(self):
        prune.remove(self.conv1, "weight")
        prune.remove(self.conv2, "weight")
        prune.remove(self.fc1, "weight")
