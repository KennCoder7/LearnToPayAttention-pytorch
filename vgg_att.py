from attention import Attention
from vgg import VGG, ConvBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
import argparse
import numpy as np


class VGG_Att(nn.Module):
    def __init__(self, pretrained_vgg: VGG, method):
        super(VGG_Att, self).__init__()
        self.__vgg = pretrained_vgg
        self.__att1 = Attention(local_shape=(256, 16, 16), global_shape=512, method=method)
        self.__att2 = Attention(local_shape=(512, 8, 8), global_shape=512, method=method)
        self.__att3 = Attention(local_shape=(512, 4, 4), global_shape=512, method=method)
        self.__classifier = nn.Sequential(
            nn.Linear(256+512+512, 10),
            nn.LogSoftmax(1))
        self.__unsample_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.__unsample_2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.__unsample_3 = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, x, return_attention_map=False):
        if not return_attention_map:
            vgg = self.__vgg(x, return_feature=True)
            att1_f = self.__att1(vgg['l1'], vgg['g'], return_attention_map=False)
            att2_f = self.__att2(vgg['l2'], vgg['g'], return_attention_map=False)
            att3_f = self.__att3(vgg['l3'], vgg['g'], return_attention_map=False)
            concat_f = torch.cat([att1_f, att2_f, att3_f], 1)
            return self.__classifier(concat_f)
        else:
            vgg = self.__vgg(x, return_feature=True)
            att1_f, att1_map = self.__att1(vgg['l1'], vgg['g'], return_attention_map=True)
            att2_f, att2_map = self.__att2(vgg['l2'], vgg['g'], return_attention_map=True)
            att3_f, att3_map = self.__att3(vgg['l3'], vgg['g'], return_attention_map=True)
            concat_f = torch.cat([att1_f, att2_f, att3_f], 1)

            att1_map = self.__unsample_1(att1_map.unsqueeze(1)).squeeze(1)
            att2_map = self.__unsample_2(att2_map.unsqueeze(1)).squeeze(1)
            att3_map = self.__unsample_3(att3_map.unsqueeze(1)).squeeze(1)

            return self.__classifier(concat_f), att1_map, att2_map, att3_map


def train(log_interval, model: VGG_Att, device, train_loader, optimizer, epoch):
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
    vgg = torch.load(r'D:\pyproject\data\LTPA-log\model\model-vgg.pkl').to("cuda:0")
    vgg_att = VGG_Att(vgg, method='pc').to("cuda:0")
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
    optimizer = optim.SGD(vgg_att.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    test_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train(100, vgg_att, "cuda:0", train_loader, optimizer, epoch)
        loss = test(vgg_att, "cuda:0", test_loader)
        if test_loss > loss:
            test_loss = loss
            if args.save_model:
                torch.save(vgg_att, r'D:\pyproject\data\LTPA-log\model\model-vgg-att.pkl')
        scheduler.step(epoch)
        visualization(nums=3)


def visualization(nums):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    vgg_att = torch.load(r'D:\pyproject\data\LTPA-log\model\model-vgg-att.pkl').to("cuda:0")
    vgg_att.eval()
    transform1 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = datasets.CIFAR10(root=r'D:\pyproject\data', train=False,
                               download=True, transform=transform1)
    test_loader = DataLoader(testset, batch_size=nums,
                             shuffle=True, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to("cuda:0")
            label = label.to("cuda:0")
            output, att1_map, att2_map, att3_map = vgg_att.forward(data, return_attention_map=True)
            pred = output.argmax(dim=1)
            for i in range(nums):
                img = data[i].cpu() * 0.5 + 0.5
                img = np.transpose(img, (1, 2, 0))
                plt.subplot(141)
                plt.imshow(img)
                plt.title('Img')
                plt.subplot(142)
                plt.imshow(img)
                plt.imshow(att1_map[i].cpu(), alpha=0.4, cmap='rainbow')
                plt.title('att1_map')
                plt.subplot(143)
                plt.imshow(img)
                plt.imshow(att2_map[i].cpu(), alpha=0.4, cmap='rainbow')
                plt.title('att2_map')
                plt.subplot(144)
                plt.imshow(img)
                plt.imshow(att3_map[i].cpu(), alpha=0.4, cmap='rainbow')
                plt.title('att3_map')
                plt.suptitle(f'Prediction={classes[pred[i].cpu()]} True={classes[label[i].cpu()]}')
                plt.show()
            break


if __name__ == '__main__':
    main()