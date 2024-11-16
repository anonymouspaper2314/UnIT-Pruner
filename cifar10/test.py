def main():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    import rtp
    from torcheval.metrics.functional import multiclass_f1_score

    class PyTorchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

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
            # x = F.relu(x)
            x = self.threshold(x)
            x = self.fc2(x)
            # x = F.relu(x)
            x = self.threshold(x)
            x = self.fc3(x)

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
        
    class RTPNoPruneNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = rtp.NPConv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = rtp.NPConv2d(6, 16, 5)
            self.fc1 = rtp.NPLinear(16 * 5 * 5, 120)
            self.fc2 = rtp.NPLinear(120, 84)
            self.fc3 = rtp.NPLinear(84, 10)

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
            # x = F.relu(x)
            x = self.threshold(x)
            x = self.fc2(x)
            # x = F.relu(x)
            x = self.threshold(x)
            x = self.fc3(x)

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
            self.conv1 = rtp.NPConv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = rtp.NPConv2d(6, 16, 5)
            self.fc1 = rtp.Linear(16 * 5 * 5, 120)
            self.fc2 = rtp.Linear(120, 84)
            self.fc3 = rtp.Linear(84, 10)

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
            # x = F.relu(x)
            x = self.threshold(x)
            x = self.fc2(x, threshold)
            # x = F.relu(x)
            x = self.threshold(x)
            x = self.fc3(x, threshold)
            
            return x
    
    class RTPDebugNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = rtp.NPConv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = rtp.NPConv2d(6, 16, 5)
            self.fc1 = rtp.DebugLinear(16 * 5 * 5, 120)
            self.fc2 = rtp.DebugLinear(120, 84)
            self.fc3 = rtp.DebugLinear(84, 10)

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
            # x = F.relu(x)
            x = self.threshold(x)
            x = self.fc2(x, threshold, stats)
            # x = F.relu(x)
            x = self.threshold(x)
            x = self.fc3(x, threshold, stats)
            
            return x

    def test_model(title, model, device, test_loader, threshold=None, debug_model=None):
        criterion = nn.CrossEntropyLoss()

        model.eval()
        test_loss = 0
        correct = 0
        total_time = 0
        f1 = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if threshold == None:
                    start_time = time.time()
                    output = model.forward(data)
                    total_time += time.time() - start_time
                else:
                    start_time = time.time()
                    output = model.forward(data, threshold)
                    total_time += time.time() - start_time

                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                f1 += multiclass_f1_score(output, target, num_classes=10).item()

        # test_loss /= len(test_loader.dataset)

        print(
            '\n{}{}:\n    Time: {:.3f}ms\n    Average loss: {:.4f}\n    F1: {:.3f}\n    Accuracy: {}/{} ({:.0f}%)'.format(
                title,
                '' if threshold == None else ' (threshold=' + str(threshold) + ')',
                total_time * 1000,
                test_loss, f1, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)
            )
        )

        if(threshold == None):
            pruned, weights = model.pruned_count()
        
            print(
                '    Average MACs Skipped in Inference: 0% (0.0000/{:.0f})\n    Average MACs Skipped in Training: {:.0f}% ({:.4f}/{})'.format(
                    weights,
                    100. * pruned / weights,
                    pruned,
                    weights
                )
            )
        else:
            stats = {
                'pruned_in_inference': 0,
                'pruned_in_training': 0,
                'total': 0
            }

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    debug_model.forward(data, threshold, stats)
                    
            print(
                '    Average MACs Skipped in Inference: {:.0f}% ({:.4f}/{})\n    Average MACs Skipped in Training: {:.0f}% ({:.4f}/{})'.format(
                    100. * stats['pruned_in_inference'] / stats['total'],
                    stats['pruned_in_inference'] / len(test_loader.dataset),
                    stats['total'] / len(test_loader.dataset),
                    100. * stats['pruned_in_training'] / stats['total'],
                    stats['pruned_in_training'] / len(test_loader.dataset),
                    stats['total'] / len(test_loader.dataset)
                )
            )

    device = torch.device('cpu')

    torch.set_num_threads(1)
    
    # UNPRUNED
    
    torch_model = PyTorchNet()
    torch_model.load_state_dict(torch.load('./cifar.pt'))
    torch_model.eval()
    
    rtp_no_prune_model = RTPNoPruneNet()
    rtp_no_prune_model.load_state_dict(torch.load('./cifar.pt'))
    rtp_no_prune_model.eval()
    
    rtp_model = RTPNet()
    rtp_model.load_state_dict(torch.load('./cifar.pt'))
    rtp_model.eval()
    
    rtp_debug_model = RTPDebugNet()
    rtp_debug_model.load_state_dict(torch.load('./cifar.pt'))
    rtp_debug_model.eval()

    # PRUNED
    
    pruned_torch_model = PyTorchNet()
    pruned_torch_model.load_state_dict(torch.load('./cifar_pruned.pt'))
    pruned_torch_model.eval()
    
    pruned_rtp_no_prune_model = RTPNoPruneNet()
    pruned_rtp_no_prune_model.load_state_dict(torch.load('./cifar_pruned.pt'))
    pruned_rtp_no_prune_model.eval()
    
    pruned_rtp_model = RTPNet()
    pruned_rtp_model.load_state_dict(torch.load('./cifar_pruned.pt'))
    pruned_rtp_model.eval()
    
    pruned_rtp_debug_model = RTPDebugNet()
    pruned_rtp_debug_model.load_state_dict(torch.load('./cifar_pruned.pt'))
    pruned_rtp_debug_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 1

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    print('UNPRUNED:\n')

    test_model('Default (PyTorch)', torch_model, device, test_loader)

    test_model('RTP (No Runtime Pruning)', rtp_no_prune_model, device, test_loader)

    threshold = 0
    while threshold <= 2:
        test_model('RTP', rtp_model, device, test_loader, threshold, rtp_debug_model)
        threshold += 0.05
    
    print('\n\nPRUNED:\n')

    test_model('Default (PyTorch)', pruned_torch_model, device, test_loader)

    test_model('RTP (No Runtime Pruning)', pruned_rtp_no_prune_model, device, test_loader)

    threshold = 0
    while threshold <= 2:
        test_model('RTP', pruned_rtp_model, device, test_loader, threshold, pruned_rtp_debug_model)
        threshold += 0.05
    
if __name__ == '__main__':
    main()