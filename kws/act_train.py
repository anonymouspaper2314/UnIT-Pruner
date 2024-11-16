def main():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.optim as optim
    from model import Net
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
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory='cpu', sampler=None)

    valid_dataset = GCommandLoader(cwd+'/data/google_speech_command/processed/valid', window_size=.02)
    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=None,
        num_workers=12, pin_memory='cpu', sampler=None)

    test_dataset = GCommandLoader(cwd+'/data/google_speech_command/processed/test', window_size=.02)
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=None,
        num_workers=12, pin_memory='cpu', sampler=None)

    def train_thresholds(model, trainloader, trainset, device):
        model.eval()
        acc = 0
        with torch.no_grad():
            for data, target in trainloader:
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
                    for data, target in trainloader:
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
                    for data, target in trainloader:
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
                for data, target in trainloader:
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
    
    torch_model = Net()
    torch_model.load_state_dict(torch.load('./kws_12.pt', weights_only=True))
    torch_model.eval()
    torch_model.to(device)

    # PRUNED
    
    pruned_torch_model = Net()
    pruned_torch_model.load_state_dict(torch.load('./kws_pruned.pt', weights_only=True))
    pruned_torch_model.eval()
    pruned_torch_model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    batch_size = 4
    
    # TRAINING UNPRUNED
    
    # [0.25, 0.5, 0.5, 0.5]
    print('Unpruned:', train_thresholds(torch_model, trainloader, train_dataset, device))
    # [0.5, 0.5, 0.5, 0.5]
    print('Pruned:', train_thresholds(pruned_torch_model, trainloader, train_dataset, device))

    torch.save(torch_model.state_dict(), './act_mnist.pt')
    torch.save(pruned_torch_model.state_dict(), './act_mnist_pruned.pt')

    
if __name__ == '__main__':
    main()