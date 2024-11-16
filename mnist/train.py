def main():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.nn.utils.prune as prune


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(256, 10)

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

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    epochs = 5
    # epochs = 0

    net = Net()

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('Unpruned...')

    acc = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
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
            for data in testloader:
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)

                # calculate outputs by running images through the network
                outputs = net(inputs)
                loss += criterion(outputs, labels)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        pruned, weights = net.pruned_count()

        acc = correct / total

        print(
            '[{:d}] acc: {:.2f}% loss: {:.3f} % weights non-zero: {:.1f}%'.format(
                epoch + 1,
                100 * correct / total,
                loss / total,
                100 * (1 - (pruned / weights))
            )
        )

    # epochs = 10

    torch.save(net.state_dict(), './mnist.pt')

    print('\nPruned...')

    last_state = net.state_dict()
    last_good_acc = acc
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epoch = 0
    while True:
        net.prune()

        for inputs, labels in trainloader:
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
            for data in testloader:
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)

                # calculate outputs by running images through the network
                outputs = net(inputs)
                loss += criterion(outputs, labels)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        pruned, weights = net.pruned_count()

        acc = correct / total

        if(last_good_acc - 0.005 > acc):
            break

        last_state = net.state_dict()


        print(
            '[{:d}] acc: {:.2f}% loss: {:.3f} % weights non-zero: {:.1f}%'.format(
                epoch + 1,
                100 * correct / total,
                loss / total,
                100 * (1 - (pruned / weights))
            )
        )
        epoch += 1
    
    net.load_state_dict(last_state)
    net.save_prune()


    print('\nFinished Training')

    torch.save(net.state_dict(), './mnist_pruned.pt')


if __name__ == '__main__':
    main()