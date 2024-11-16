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
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.thresh1 = nn.Threshold(0, 0)
            self.pool = nn.MaxPool2d(4, 4)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.thresh2 = nn.Threshold(0, 0)
            self.fc1 = nn.Linear(288, 12)


        def forward(self, x):
            x = self.conv1(x)
            x = self.thresh1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.thresh2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.fc1(x)

            return x
        
        def set_thresholds(self, thresholds):
            self.thresh1 = nn.Threshold(thresholds[0], 0)
            self.thresh2 = nn.Threshold(thresholds[1], 0)
        
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
            self.thresh1 = nn.Threshold(0, 0)
            self.pool = nn.MaxPool2d(4, 4)
            self.conv2 = rtp.NPConv2d(6, 16, 5)
            self.thresh2 = nn.Threshold(0, 0)
            self.fc1 = rtp.NPLinear(288, 12)


        def forward(self, x):
            x = self.conv1(x)
            x = self.thresh1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.thresh2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.fc1(x)

            return x
        
        def set_thresholds(self, thresholds):
            self.thresh1 = nn.Threshold(thresholds[0], 0)
            self.thresh2 = nn.Threshold(thresholds[1], 0)
        
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
        
    class RTPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = rtp.Conv2d(1, 6, 5)
            self.thresh1 = nn.Threshold(0, 0)
            self.pool = nn.MaxPool2d(4, 4)
            self.conv2 = rtp.Conv2d(6, 16, 5)
            self.thresh2 = nn.Threshold(0, 0)
            self.fc1 = rtp.Linear(288, 12)

        def forward(self, x, threshold):
            x = self.conv1(x, threshold)
            x = self.thresh1(x)
            x = self.pool(x)
            x = self.conv2(x, threshold)
            x = self.thresh2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.fc1(x, threshold)

            return x
        
        def set_thresholds(self, thresholds):
            self.thresh1 = nn.Threshold(thresholds[0], 0)
            self.thresh2 = nn.Threshold(thresholds[1], 0)

    
    class RTPDebugNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = rtp.DebugConv2d(1, 6, 5)
            self.thresh1 = nn.Threshold(0, 0)
            self.pool = nn.MaxPool2d(4, 4)
            self.conv2 = rtp.DebugConv2d(6, 16, 5)
            self.thresh2 = nn.Threshold(0, 0)
            self.fc1 = rtp.DebugLinear(288, 12)

        def forward(self, x, threshold, stats):
            x = self.conv1(x, threshold, stats)
            x = self.thresh1(x)
            x = self.pool(x)
            x = self.conv2(x, threshold, stats)
            x = self.thresh2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.fc1(x, threshold, stats)

            return x
        
        def set_thresholds(self, thresholds):
            self.thresh1 = nn.Threshold(thresholds[0], 0)
            self.thresh2 = nn.Threshold(thresholds[1], 0)

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
                f1 += multiclass_f1_score(output, target, num_classes=12).item()

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
    
    # threshs = [0, 0]
    threshs = [0.5, 0.25]
    
    torch_model = PyTorchNet()
    torch_model.load_state_dict(torch.load('./kws_12.pt', map_location='cpu', weights_only=False))
    torch_model.eval()
    torch_model.set_thresholds(threshs)
    
    rtp_no_prune_model = RTPNoPruneNet()
    rtp_no_prune_model.load_state_dict(torch.load('./kws_12.pt', map_location='cpu', weights_only=False))
    rtp_no_prune_model.eval()
    rtp_no_prune_model.set_thresholds(threshs)
    
    rtp_model = RTPNet()
    rtp_model.load_state_dict(torch.load('./kws_12.pt', map_location='cpu', weights_only=False))
    rtp_model.eval()
    rtp_model.set_thresholds(threshs)
    
    rtp_debug_model = RTPDebugNet()
    rtp_debug_model.load_state_dict(torch.load('./kws_12.pt', map_location='cpu', weights_only=False))
    rtp_debug_model.eval()
    rtp_debug_model.set_thresholds(threshs)

    # PRUNED
    
    # pruned_threshs = [0, 0]
    pruned_threshs = [0.5, 0.5]
    
    pruned_torch_model = PyTorchNet()
    pruned_torch_model.load_state_dict(torch.load('./kws_pruned.pt', map_location='cpu', weights_only=False))
    pruned_torch_model.eval()
    pruned_torch_model.set_thresholds(pruned_threshs)
    
    pruned_rtp_no_prune_model = RTPNoPruneNet()
    pruned_rtp_no_prune_model.load_state_dict(torch.load('./kws_pruned.pt', map_location='cpu', weights_only=False))
    pruned_rtp_no_prune_model.eval()
    pruned_rtp_no_prune_model.set_thresholds(pruned_threshs)
    
    pruned_rtp_model = RTPNet()
    pruned_rtp_model.load_state_dict(torch.load('./kws_pruned.pt', map_location='cpu', weights_only=False))
    pruned_rtp_model.eval()
    pruned_rtp_model.set_thresholds(pruned_threshs)
    
    pruned_rtp_debug_model = RTPDebugNet()
    pruned_rtp_debug_model.load_state_dict(torch.load('./kws_pruned.pt', map_location='cpu', weights_only=False))
    pruned_rtp_debug_model.eval()
    pruned_rtp_debug_model.set_thresholds(pruned_threshs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 1
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.nn.utils.prune as prune
    from dataset import SpeechCommands
    from model import Net
    from data_loader import GCommandLoader
    import numpy as np
    import data
    import os
    from tqdm import tqdm
    from sklearn.metrics import f1_score

    ninp = 80
    nout = 12

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(23)
    np.random.seed(23)
    batch_size  = 16

    cwd = os.getcwd()
    train_dataset = GCommandLoader(cwd+'/data/google_speech_command/processed/train', window_size=.02)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory='cpu', sampler=None)

    valid_dataset = GCommandLoader(cwd+'/data/google_speech_command/processed/valid', window_size=.02)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=None,
        num_workers=12, pin_memory='cpu', sampler=None)

    test_dataset = GCommandLoader(cwd+'/data/google_speech_command/processed/test', window_size=.02)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=None,
        num_workers=12, pin_memory='cpu', sampler=None)

    print('UNPRUNED:\n')

    test_model('Default (PyTorch)', torch_model, device, test_loader)

    test_model('RTP (No Runtime Pruning)', rtp_no_prune_model, device, test_loader)

    threshold = 0
    while threshold <= 0:
        test_model('RTP', rtp_model, device, test_loader, threshold, rtp_debug_model)
        threshold += 0.05
    
    print('\n\nPRUNED:\n')

    test_model('Default (PyTorch)', pruned_torch_model, device, test_loader)

    test_model('RTP (No Runtime Pruning)', pruned_rtp_no_prune_model, device, test_loader)

    threshold = 0
    while threshold <= 1:
        test_model('RTP', pruned_rtp_model, device, test_loader, threshold, pruned_rtp_debug_model)
        threshold += 0.05
    
if __name__ == '__main__':
    main()