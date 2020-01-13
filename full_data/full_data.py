import torch

from general.helpers import get_accuracy
from general.models import CNN
from general.train import train_model
import pickle

data_objects_root = "full_data/data_objects/"
training_data_obj = open(data_objects_root + "training_data_obj", "rb")
validation_data_obj = open(data_objects_root + "validation_data_obj", "rb")
testing_data_obj = open(data_objects_root + "testing_data_obj", "rb")

training_data = pickle.load(training_data_obj)
validation_data = pickle.load(validation_data_obj)
testing_data = pickle.load(testing_data_obj)

network = CNN()
train_save_directory = "full_data/results/"
model_save_directory = "full_data/models/"

train = False
if train:
    train_model(network, "CNN", training_data, batch_size=256, epoch_count=150, shuffle=True, learning_rate=0.0005,
                checkpoint_frequency=20, train_save=train_save_directory, model_save=model_save_directory)

test = True
if test:
    state = torch.load('full_data/models/CNN_860_256_0.0005')
    network.load_state_dict(state)
    accuracy = get_accuracy(network, testing_data)
    print("Testing Accuracy = {}%".format(round(accuracy * 100, 2)))
