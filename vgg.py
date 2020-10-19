import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchsummary import summary
import argparse


class ConvBlock(nn.Module):
    def __init__(self, i_dm, o_dm, k_s, pool=False):
        super(ConvBlock, self).__init__()
        layer = [
            nn.Conv2d(in_channels=i_dm, out_channels=o_dm, kernel_size=k_s, padding=1),
            nn.BatchNorm2d(num_features=o_dm),
            nn.ReLU()
        ]
        if pool:
            layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layer)

    def forward(self, x):
        return self.block(x)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.Conv1 = nn.Sequential(
            ConvBlock(3, 64, 3),
            ConvBlock(64, 128, 3),
            ConvBlock(128, 256, 3, pool=True),
        )
        self.Conv2 = nn.Sequential(
            ConvBlock(256, 512, 3, pool=True),
        )
        self.Conv3 = nn.Sequential(
            ConvBlock(512, 512, 3, pool=True),
        )
        self.Conv4 = nn.Sequential(
            ConvBlock(512, 512, 3, pool=True),
            ConvBlock(512, 512, 3, pool=True),
        )
        self.L = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.Classifier = nn.Sequential(
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, return_feature=False):
        c1 = self.Conv1(x)
        c2 = self.Conv2(c1)
        c3 = self.Conv3(c2)
        c4 = self.Conv4(c3)
        flatten = torch.flatten(c4, 1)
        gbf = self.L(flatten)
        if return_feature:
            return {'g': gbf, 'l1': c1, 'l2': c2, 'l3': c3}
        return self.Classifier(gbf)


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--save-model', type=bool, default=True, metavar='N',
                    help='whether save model')
args = parser.parse_args()


def main():
    cnn = VGG().to("cuda:0")
    summary(cnn, (3, 32, 32), args.batch_size)
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform1 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=r'D:\pyproject\data', train=True,
                                download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    testset = datasets.CIFAR10(root=r'D:\pyproject\data', train=False,
                               download=True, transform=transform1)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    torch.manual_seed(7)
    optimizer = optim.SGD(cnn.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    test_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train(100, cnn, "cuda:0", train_loader, optimizer, epoch)
        loss = test(cnn, "cuda:0", test_loader)
        if test_loss > loss:
            test_loss = loss
            if args.save_model:
                torch.save(cnn, r'D:\pyproject\data\LTPA-log\model\model-vgg.pkl')
        scheduler.step(epoch)


def reload():
    cnn = torch.load(r'D:\pyproject\data\LTPA-log\model\model-vgg.pkl').to("cuda:0")
    summary(cnn, (3, 32, 32), args.batch_size)
    transform1 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = datasets.CIFAR10(root=r'D:\pyproject\data', train=False,
                               download=True, transform=transform1)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    test(cnn, "cuda:0", test_loader)


if __name__ == '__main__':
    main()
    reload()

