# Imports:
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tensorflow as tf
from torchvision.transforms import ToTensor
from io import BytesIO
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

# Creating Data:
letter_array = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
std_no = '1002951754'
image_string = '_A_1.jpg'
images = []
labels = []
for letter_index in range(1, 10):
    label = torch.tensor(letter_index - 1)
    for letter_sample in range(1, 4):
        image = Image.open("small_data/personal_data/" + std_no + image_string)
        image = ToTensor()(image)
        images.append(image)
        labels.append(label)
        if letter_sample == 3:
            image_string = image_string.replace(str(letter_sample), str(1))
        else:
            image_string = image_string.replace(str(letter_sample), \
                                                str(letter_sample + 1))
    if letter_index == 9:
        break
    image_string = image_string.replace(letter_array[letter_index - 1], \
                                        letter_array[letter_index])


class Dataset():
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]
        return x, y


small_data = Dataset(images, labels)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 18, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(18, 24, kernel_size=7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        hidden_layer_size = 128
        hidden_layer_2_size = 64
        self.fc1 = nn.Linear(24 * 10 * 10, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_2_size)
        self.fc3 = nn.Linear(hidden_layer_2_size, 9)

    def forward(self, x):
        # The input x is a tensor of the following dimension:
        # [batch_size, RGB, H, W]
        # [batch_size, 3, 224, 224]
        # General formula for new H, W using a kernel with size k, padding p,
        # and stride s:
        # H = (Hinit - k + p)/s + 1

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 24 * 10 * 10)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return (x)


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




network = CNN()
train(network, 'small_data', small_data, None, num_epochs=100, learning_rate=0.001, batch_size=27, \
      checkpoint_frequency=25, save=False)
