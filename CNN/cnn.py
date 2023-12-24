import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from medmnist import BreastMNIST
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)  # Flatten before fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set hyperparameter
EPOCH = 100
pre_epoch = 0
batch_size = 128
initial_lr = 0.001

# prepare dataset and preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4465], [0.2010])
])

transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4465], [0.2010])
])

# BreastMNIST dataset
train_dataset = BreastMNIST(split="train", download=True, transform=transform_train)
val_dataset = BreastMNIST(split="val", download=True, transform=transform_test)
test_dataset = BreastMNIST(split="test", download=True, transform=transform_test)

classes = ('positive', 'negative')

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# define CNN
net = CNN().to(device)

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

    # get the accuracy with train_dataset in each epoch
    # print('Waiting Test...')
    with torch.no_grad():
        accuracy_test = 0
        total = 0
        for data in test_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            labels = labels.squeeze().long()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy_test += (outputs.argmax(1) == labels).sum()
            for i in range(len(predicted)):
                print(f"Image {i + 1}: Predicted - {classes[predicted[i]]}, Actual - {classes[labels[i]]}")

        print('Test\'s accuracy is: %.3f%%' % (100 * accuracy_test / total))

    scheduler.step()

print('Train has finished, total epoch is %d' % EPOCH)
