def main():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.optim as optim
    from preprocessing import HARdataset

    class PyTorchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 6, 5)
            self.thresh1 = nn.Threshold(0, 0)
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(6, 16, 5)
            self.thresh2 = nn.Threshold(0, 0)
            self.fc1 = nn.Linear(112, 10)


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
    
    def train_thresholds(model, trainloader, trainset, device):
        model.eval()
        acc = 0
        with torch.no_grad():
            for data, target, _ in trainloader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                acc += pred.eq(target.view_as(pred)).sum().item()
        acc /= (len(trainset) * 0.8)
        thresholds = [0, 0]
        for i in range(1, -1, -1):
            speed = 1
            thresholds[i] = 1
            while True:
                model.set_thresholds(thresholds)

                test_acc = 0
                with torch.no_grad():
                    for data, target, _ in trainloader:
                        data = data.to(device)
                        target = target.to(device)

                        output = model(data)

                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        test_acc += pred.eq(target.view_as(pred)).sum().item()
                test_acc /= (len(trainset) * 0.8)

                if(test_acc > acc - 0.002):
                    thresholds[i] += speed
                    speed *= 2
                else:
                    break
            
            print(thresholds)
            speed /= 2
            thresholds[i] -= speed
            speed /= 2
            
            while True:
                model.set_thresholds(thresholds)

                test_acc = 0
                with torch.no_grad():
                    for data, target, _ in trainloader:
                        data = data.to(device)
                        target = target.to(device)

                        output = model(data)

                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        test_acc += pred.eq(target.view_as(pred)).sum().item()
                test_acc /= (len(trainset) * 0.8)

                print(thresholds)
                if(test_acc < acc - 0.002):
                    thresholds[i] -= speed
                    speed /= 2
                elif(test_acc > acc - 0.0025):
                    break
                else:
                    thresholds[i] += speed
                    speed /= 2

            acc = 0
            with torch.no_grad():
                for data, target, _ in trainloader:
                    data = data.to(device)
                    target = target.to(device)

                    output = model(data)

                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    acc += pred.eq(target.view_as(pred)).sum().item()
            acc /= (len(trainset) * 0.8)

        return thresholds

    device = torch.device('cuda')

    torch.set_num_threads(1)
    
    # UNPRUNED
    
    torch_model = PyTorchNet()
    torch_model.load_state_dict(torch.load('./har.pt', weights_only=True))
    torch_model.eval()
    torch_model.to(device)

    # PRUNED
    
    pruned_torch_model = PyTorchNet()
    pruned_torch_model.load_state_dict(torch.load('./har_pruned.pt', weights_only=True))
    pruned_torch_model.eval()
    pruned_torch_model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    batch_size = 4
    
    dataset = HARdataset('./data/pml-training.csv')
    train_sampler, test_sampler = dataset.split_ind(val_split=0.2, shuffle=False)
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2,sampler=test_sampler)
    
    # TRAINING UNPRUNED
    
    # [0.25, 0.5, 0.5, 0.5]
    print('Unpruned:', train_thresholds(torch_model, trainloader, dataset, device))
    # [0.5, 0.5, 0.5, 0.5]
    print('Pruned:', train_thresholds(pruned_torch_model, trainloader, dataset, device))

    torch.save(torch_model.state_dict(), './act_mnist.pt')
    torch.save(pruned_torch_model.state_dict(), './act_mnist_pruned.pt')

    
if __name__ == '__main__':
    main()