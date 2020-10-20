import torch
import pickle

from general.consts import DirectoryConsts, FileNames, TrainingConsts, TestingConsts
from general.helpers import get_accuracy
from general.models import CNN
from general.train import train_model
from general.test import test_model

training_data_obj = open(DirectoryConsts.dataset_root + FileNames.training_data_obj, "rb")
validation_data_obj = open(DirectoryConsts.dataset_root + FileNames.validation_data_obj, "rb")
testing_data_obj = open(DirectoryConsts.dataset_root + FileNames.testing_data_obj, "rb")

training_data = pickle.load(training_data_obj)
validation_data = pickle.load(validation_data_obj)
testing_data = pickle.load(testing_data_obj)

network = CNN()

train = TrainingConsts.train
if train:
    train_model(network, "CNN", training_data, DirectoryConsts.results_directory,
                DirectoryConsts.model_save_directory, validation_data=validation_data,
                batch_size=TrainingConsts.batch_size, epoch_count=TrainingConsts.epoch_count,
                shuffle=TrainingConsts.shuffle_flag, learning_rate=TrainingConsts.learning_rate,
                checkpoint_frequency=TrainingConsts.checkpoint_frequency, momentum=TrainingConsts.momentum,
                scheduler=TrainingConsts.scheduler, save=TrainingConsts.save_flag)

test = TestingConsts.test
if test:
    outputs = test_model(network, DirectoryConsts.model_save_directory, TestingConsts.test_model_name, testing_data)
