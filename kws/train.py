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

device = torch.device('cuda')

log_file = open("scores.log", "w")
def main():    
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 5
    print('Unpruned...')

    for epoch in range(epochs):
        for inputs, labels in tqdm(train_loader):
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
        test_y = []
        pred_test_y = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)

                # calculate outputs by running images through the network
                outputs = net(inputs)
                loss += criterion(outputs, labels)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_y = test_y + list(labels.detach().cpu().numpy())
                pred_test_y = pred_test_y + list(predicted.detach().cpu().numpy())
        
        pruned, weights = net.pruned_count()
        f1= f1_score(y_true=test_y, y_pred=pred_test_y, average='macro')
        log_file.write('[{:d}] acc: {:.2f}% loss: {:.3f} % weights non-zero: {:.1f} % f1:{:.2f} % \n'.format(
                epoch + 1,
                100 * correct / total,
                loss / total,
                100 * (1 - (pruned / weights)),
                100* f1
            ))
        

        print(
            '[{:d}] acc: {:.2f}% loss: {:.3f} % weights non-zero: {:.1f} % f1:{:.2f} %'.format(
                epoch + 1,
                100 * correct / total,
                loss / total,
                100 * (1 - (pruned / weights)),
                100* f1
            )
        )

    epochs = 25

    print('\nPruned...')

    net_pruned = Net()

    net_pruned.to(device)

    optimizer = optim.SGD(net_pruned.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        net_pruned.prune()

        for inputs, labels in tqdm(train_loader):
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
        test_y = []
        pred_test_y = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in tqdm(test_loader):
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)

                # calculate outputs by running images through the network
                outputs = net_pruned(inputs)
                loss += criterion(outputs, labels)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_y = test_y + list(labels.detach().cpu().numpy())
                pred_test_y = pred_test_y + list(predicted.detach().cpu().numpy())
        
        pruned, weights = net_pruned.pruned_count()
        f1= f1_score(y_true=test_y, y_pred=pred_test_y, average='macro')
        log_file.write('[{:d}] acc: {:.2f}% loss: {:.3f} % weights non-zero: {:.1f} % f1:{:.2f} % \n'.format(
                epoch + 1,
                100 * correct / total,
                loss / total,
                100 * (1 - (pruned / weights)),
                100* f1
            ))
        print(
            '[{:d}] acc: {:.2f}% loss: {:.3f} % weights non-zero: {:.1f}% f1: {:.2f}%'.format(
                epoch + 1,
                100 * correct / total,
                loss / total,
                100 * (1 - (pruned / weights)),
                100*f1
            )
        )
    
    net_pruned.save_prune()

    log_file.close()
    print('\nFinished Training')

    torch.save(net.state_dict(), './kws_12.pt')

    torch.save(net_pruned.state_dict(), './kws_pruned.pt')


if __name__ == '__main__':
    main()