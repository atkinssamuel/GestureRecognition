import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal
from general.helpers import get_accuracy, get_loss
from general.plotting import plot_train_valid, plot_train


def train_model(model, name, training_data, results_dir, model_save, validation_data=None, batch_size=1, epoch_count=1,
                shuffle=False, learning_rate=0.01, checkpoint_frequency=5, momentum=0.9, scheduler=False, save=False):

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # The scheduler reduces the learning rate when the loss begins to plateau:
    if scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    iterations, losses, train_acc, validation_acc, validation_loss = [], [], [], [], []

    # training
    current_iteration = 0  # the number of iterations

    for epoch in range(epoch_count):
        for inputs, labels in iter(train_loader):
            optimizer.zero_grad()  # a clean up step for PyTorch
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            if scheduler:
                scheduler.step()

            # save the current training information
            iterations.append(current_iteration)
            losses.append(float(loss) / batch_size)  # compute *average* loss
            train_acc.append(get_accuracy(model, training_data))  # compute training accuracy

            if validation_data is not None:
                validation_acc.append(get_accuracy(model, validation_data))
                validation_loss.append(get_loss(model, validation_data))

            # checkpoint:
            if current_iteration % checkpoint_frequency == 0:
                print("Training Accuracy at Iteration {}: {}".format(current_iteration, train_acc[-1]))
                print("Training Loss at Iteration {}: {}".format(current_iteration, losses[-1]))
                print("Learning Rate at Iteration {}: {}\n".format(current_iteration, optimizer.param_groups[0]['lr']))
                if save:
                    model_string = str(name) + '_' + str(current_iteration) + '_' + str(batch_size) + \
                                   '_' + str(learning_rate)
                    torch.save(model.state_dict(), model_save + model_string)

            current_iteration += 1
        # scheduler.step(get_loss(model, training_data))

    # plotting
    if validation_data is not None:
        plot_train_valid(iterations, validation_acc, train_acc, validation_loss, losses, results_dir)
    else:
        plot_train(iterations, train_acc, losses, results_dir)






    print("Final Training Accuracy: {}".format(train_acc[-1]))