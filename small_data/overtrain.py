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

from main.models import CNN
from main.train import train

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
            image_string = image_string.replace(str(letter_sample),
                                                str(letter_sample + 1))
    if letter_index == 9:
        break
    image_string = image_string.replace(letter_array[letter_index - 1],
                                        letter_array[letter_index])


class Data():
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]
        return x, y


small_data = Data(images, labels)
network = CNN()
train(network, 'small_data', small_data, None, num_epochs=100, learning_rate=0.001, batch_size=27,
      checkpoint_frequency=25, save=False, train_save="small_data/results/overtrain_training")
