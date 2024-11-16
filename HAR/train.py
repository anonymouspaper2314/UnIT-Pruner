def main():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.nn.utils.prune as prune
    from torch.utils.data import random_split
    from preprocessing import HARdataset


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 6, 5)
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(6, 16, 5)
            self.fc1 = nn.Linear(112, 10)

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

    device = torch.device('cuda')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    batch_size = 4

    torch.manual_seed(42)

    train_dataset, test_dataset = random_split(HARdataset('./data/pml-training.csv'), [0.8, 0.2])
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2)    

    epochs = 15

    net = Net()

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('Unpruned...')

    for epoch in range(epochs):
        for inputs, labels, _ in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        correct = 0
        total = 0
        loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for inputs, labels, _ in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # calculate outputs by running images through the network
                outputs = net(inputs)
                loss += criterion(outputs, labels)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        pruned, weights = net.pruned_count()

        print(
            '[{:d}] acc: {:.2f}% loss: {:.3f} % weights non-zero: {:.1f}%'.format(
                epoch + 1,
                100 * correct / total,
                loss / total,
                100 * (1 - (pruned / weights))
            )
        )

    epochs = 25

    print('\nPruned...')

    net_pruned = Net()

    net_pruned.to(device)

    optimizer = optim.SGD(net_pruned.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        net_pruned.prune()

        for inputs, labels, _ in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net_pruned(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        correct = 0
        total = 0
        loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                inputs, labels, _ = data

                inputs, labels = inputs.to(device), labels.to(device)

                # calculate outputs by running images through the network
                outputs = net_pruned(inputs)
                loss += criterion(outputs, labels)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        pruned, weights = net_pruned.pruned_count()

        print(
            '[{:d}] acc: {:.2f}% loss: {:.3f} % weights non-zero: {:.1f}%'.format(
                epoch + 1,
                100 * correct / total,
                loss / total,
                100 * (1 - (pruned / weights))
            )
        )
    
    net_pruned.save_prune()


    print('\nFinished Training')

    torch.save(net.state_dict(), './har.pt')

    torch.save(net_pruned.state_dict(), './har_pruned.pt')


if __name__ == '__main__':
    main()