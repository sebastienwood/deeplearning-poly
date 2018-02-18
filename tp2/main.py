import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from torch import nn
from torch import optim
from torchvision.datasets import FashionMNIST
from torch.autograd import Variable

train_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

valid_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)
train_data.train_data = train_data.train_data[train_idx, :]
train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]

mask = np.ones(60000)
mask[train_idx] = 0

valid_data.train_data = valid_data.train_data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.train_labels = valid_data.train_labels[torch.from_numpy(mask).type(torch.ByteTensor)]

batch_size = 100
test_batch_size = 100

train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_data,
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    FashionMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


class FcNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=0)
        return x

class FcNetwork2(nn.Module):
    # First improvement tried switch to Relu in HL
    # Around 6% perf improvement all others things equals
    # 83%
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=0)
        return x

class FcNetwork3(nn.Module):
    # Tried a Convolutional Layer instead (because we're using images)
    # 1 layer of convolutional (size 28) without maxpool, using leakyrelu
    # 2 fc
    # solid 91% - going for 92% in 20 epochs (stuck around that point)


    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv2d(1, 28, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(4032, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.leaky_relu(self.cv1(x))
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=0)
        return x

class FcNetwork4(nn.Module):
    # Improved version with 2 conv and maxpool
    # Results surprisingly low around 89% in 15 generations

    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.leaky_relu(self.cv1(x))
        x = self.mp(x)
        x = F.leaky_relu(self.cv2(x))
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=0)
        return x


model = FcNetwork3()
optimizer = optim.Adam(model.parameters())

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

ll = []
def test(loader, name):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    ll.append(test_loss)
    print('\n' + name + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

epochs = 50

for epoch in range(1, epochs + 1):
    train(epoch)
    test(valid_loader, 'valid')

test(test_loader, 'test')
np.savetxt('fcnetwork3', ll, delimiter=',')
