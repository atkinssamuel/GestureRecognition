# Imports:
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from general.models import CNN
from general.helpers import Data
from general.train import train_model

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

small_data = Data(images, labels)

network = CNN()
train_save_directory = "small_data/results/"
model_save_directory = "small_data/models/"
train_model(network, "CNN", small_data, batch_size=256, epoch_count=100, shuffle=True, learning_rate=0.001,
            checkpoint_frequency=20, train_save=train_save_directory, model_save=model_save_directory)
