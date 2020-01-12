# Imports:
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler


def get_accuracy(model, data):
    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64):
        output = model(imgs)
        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
        return correct / total


def get_loss(model, data):
    data_loader = torch.utils.data.DataLoader(data, batch_size=data.__len__(), shuffle=False)
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in iter(data_loader):
        out = model(imgs)  # forward pass
        loss = criterion(out, labels)
        return float(loss) / data.__len__()
