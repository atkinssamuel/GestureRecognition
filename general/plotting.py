import matplotlib.pyplot as plt
import scipy
import numpy as np

def plot_train_valid(iterations, validation_acc, train_acc, validation_loss, train_loss, results_dir):
    plt.title("Training and Validation Accuracies")
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(validation_acc), polyorder=3, window_length=5),
             label="Validation")
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(train_acc), polyorder=3, window_length=5),
             label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(results_dir + "training_valid_accuracy.png")
    plt.close()

    plt.title("Training and Validation Losses")
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(validation_loss), polyorder=3, window_length=5),
             label="Validation")
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(train_loss), polyorder=3, window_length=5),
             label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(results_dir + "training_valid_loss.png")
    plt.close()
    return

def plot_train(iterations, train_acc, train_loss, results_dir):
    plt.title("Training Accuracy")
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(train_acc), polyorder=3, window_length=5),
             label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(results_dir + "training_accuracy.png")
    plt.close()

    plt.title("Training Loss")
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(train_loss), polyorder=3, window_length=5),
             label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.savefig(results_dir + "training_loss.png")
    plt.close()