# Imports:
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from main.helpers import get_accuracy, get_loss


def train(model, name, data, validation_data, batch_size=1, num_epochs=1, shuffle=False, learning_rate=0.01,
          checkpoint_frequency=10, save=True, train_save="default.png", valid_save="default.png"):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    iters, losses, train_acc, validation_acc, validation_loss = [], [], [], [], []

    # training
    current_iteration = 0  # the number of iterations

    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            out = model(imgs)  # forward pass
            loss = criterion(out, labels)  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch

            # save the current training information
            iters.append(current_iteration)
            losses.append(float(loss) / batch_size)  # compute *average* loss
            train_acc.append(get_accuracy(model, data))  # compute training accuracy
            if validation_data is not None:
                validation_acc.append(get_accuracy(model, validation_data))
                validation_loss.append(get_loss(model, validation_data))

            # checkpoint:
            if current_iteration % checkpoint_frequency == 0:
                print("Current Training Accuracy at Iteration {}: {}" \
                      .format(current_iteration, train_acc[-1]))
                if save:
                    model_path = str(name) + '_' + str(current_iteration) + '_' + \
                                 str(batch_size) + '_' + str(learning_rate)
                    torch.save(model.state_dict(), model_path)

            current_iteration += 1

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(train_save + "_loss.png")
    plt.show()
    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.savefig(train_save + "_accuracy.png")
    plt.show()

    if validation_data is not None:
        plt.title("Validation Accuracy")
        plt.plot(validation_acc, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc='best')
        plt.savefig(valid_save + "_accuracy.png")
        plt.show()
        plt.title("Validation Loss")
        plt.plot(validation_loss, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Validation Loss")
        plt.legend(loc='best')
        plt.savefig(valid_save + "_loss.png")
        plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
