# -*- coding: utf-8 -*
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(
            # x:[b,3,32,32] => [b,6,28,28] = > [b,6,14,14]
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # x:[b,6,14,14] => [b,16,10,10] => [b,16,5,5]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        batchsize = x.size(0)
        # x:[b,3,32,32] => [b,16,5,5]
        x = self.conv(x)
        # x:[b,16,5,5] => [b,16*5*5]
        x = x.view(batchsize, 16*5*5)
        # [b,16*5*5] => [b,10]
        logits = self.fc(x)
        return logits


def main():
    batchsize = 16
    epochs = 2
    # lenet = LeNet5()
    lenet = torch.load(sys.path[0]+"./lenet.pt")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lenet.parameters(), lr=1e-3, momentum=0.9)

    CIFAR_TRAIN_SET = torchvision.datasets.CIFAR10(sys.path[0]+'./CIFAR10',
                                                   train=True,
                                                   transform=transforms.Compose([
                                                       transforms.Resize(
                                                           (32, 32)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(
                                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                   ]),
                                                   download=False)
    CIFAR_TEST_SET = torchvision.datasets.CIFAR10(sys.path[0]+'./CIFAR10',
                                                  train=False,
                                                  transform=transforms.Compose([
                                                      transforms.Resize(
                                                          (32, 32)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(
                                                          (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                  ]),
                                                  download=False)
    CIFAR_TRAIN = torch.utils.data.DataLoader(
        dataset=CIFAR_TRAIN_SET, batch_size=batchsize, shuffle=True, num_workers=2)
    CIFAR_TEST = torch.utils.data.DataLoader(
        dataset=CIFAR_TEST_SET, batch_size=batchsize, shuffle=True, num_workers=2)

    print("训练开始：")
    stime = time.time()
    for epoch in range(epochs):
        loss = 0.0
        for i, (x, label) in enumerate(CIFAR_TRAIN):
            logits = lenet(x)
            loss = criterion(logits, label)  # loss: tensor scalar
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch  %d  loss:  %.3f" % (epoch+1, loss))
        torch.save(lenet, './lenet.pt')

    etime = time.time()
    print("训练结束，耗费%.2f秒" % (etime-stime))

    classes = CIFAR_TRAIN_SET.classes
    correct = [0. for i in range(10)]
    total = [0. for i in range(10)]
    with torch.no_grad():
        for index, (x, label) in enumerate(CIFAR_TEST):
            logits = lenet(x)
            # 返回的是dim=1时，[0]代表最大值，[1]是该最大值所在索引，第一个变量没用
            _, predicted = torch.max(logits.data, 1)
            res = (predicted == label).squeeze()
            for i in range(batchsize):
                total[label[i].item()] += 1
                correct[label[i].item()] += res[i].item()

    for i in range(10):
        print("%-10s有 %d 张，识别成功了 %d 张，准确率 %.1f%%" %
              (classes[i], total[i], correct[i], 100 * correct[i] / total[i]))


if __name__ == "__main__":
    main()
