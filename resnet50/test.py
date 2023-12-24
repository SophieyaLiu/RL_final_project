# Use the ResNet50 on MedMNIST
from resnet50test import Bottleneck
from resnet50test import ResNet
from medmnist import BreastMNIST
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler

# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set hyperparameter
EPOCH = 100
pre_epoch = 0
batch_size = 128
initial_lr = 0.001

# BreastMNIST dataset
train_dataset = BreastMNIST(split="train", download=True, transform=torchvision.transforms.ToTensor())
val_dataset = BreastMNIST(split="val", download=True, transform=torchvision.transforms.ToTensor())
test_dataset = BreastMNIST(split="test", download=True, transform=torchvision.transforms.ToTensor())

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# labels in Resnet
classes = ('positive', 'negative')

# define ResNet50
net = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)

# define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

for epoch in range(pre_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(train_loader, 0):
        # prepare dataset
        length = len(train_loader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # forward & backward
        outputs = net(inputs)
        labels = labels.squeeze().long()

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print ac & loss in each batch
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

    # get the ac with test_dataset in each epoch
    #print('Waiting Test...')
    with torch.no_grad():
        accuracy_test = 0
        val_len = len(test_dataset)
        for data in test_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze().long()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy_test += (outputs.argmax(1) == labels).sum()
            for i in range(len(predicted)):
                print(f"Image {i + 1}: Predicted - {classes[predicted[i]]}, Actual - {classes[labels[i]]}")
        print('Test\'s accuracy is: %.3f%%' % (100. * accuracy_test / val_len))
    scheduler.step()
print('Train has finished, total epoch is %d' % EPOCH)
