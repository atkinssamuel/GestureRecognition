# Imports:
import pickle

from PIL import Image
from torchvision.transforms import ToTensor

from general.helpers import Data
from general.consts import DirectoryConsts

# There are 102 student samples for each letter available in the dataset
# The dataset files are in the following format:
# *student_number* + _ + *letter* + _ + *image_index*

letter_array = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# Defining data arrays:
training_images = []
training_labels = []
validation_images = []
validation_labels = []
testing_images = []
testing_labels = []

for letter in letter_array:
    print("Adding images for letter {}".format(letter))
    for student_number in range(1, 75):
        for image_index in range(1, 4):
            image_string = str(letter) + '/' + str(student_number) + '_' + letter + \
                           '_' + str(image_index) + '.jpg'
            try:
                image = Image.open(DirectoryConsts.dataset_root + image_string)
            except:
                continue
            image = ToTensor()(image)
            if list(image.shape) != list([3, 224, 224]):
                continue
            training_images.append(image)
            label = letter_array.index(letter)
            training_labels.append(label)

    for student_number in range(75, 90):
        for image_index in range(1, 4):
            image_string = str(letter) + '/' + str(student_number) + '_' + letter + \
                           '_' + str(image_index) + '.jpg'
            try:
                image = Image.open(DirectoryConsts.dataset_root + image_string)
            except:
                continue
            image = ToTensor()(image)
            if list(image.shape) != list([3, 224, 224]):
                continue
            validation_images.append(image)
            label = letter_array.index(letter)
            validation_labels.append(label)

    for student_number in range(90, 103):
        for image_index in range(1, 4):
            image_string = str(letter) + '/' + str(student_number) + '_' + letter + \
                           '_' + str(image_index) + '.jpg'
            try:
                image = Image.open(DirectoryConsts.dataset_root + image_string)
            except:
                continue
            image = ToTensor()(image)
            if list(image.shape) != list([3, 224, 224]):
                continue
            testing_images.append(image)
            label = letter_array.index(letter)
            testing_labels.append(label)

training_data = Data(training_images, training_labels)
validation_data = Data(validation_images, validation_labels)
testing_data = Data(testing_images, testing_labels)

data_objects_root = "full_data/data_objects/"

training_data_obj = open(data_objects_root + "training_data_obj", "wb")
validation_data_obj = open(data_objects_root + "validation_data_obj", "wb")
testing_data_obj = open(data_objects_root + "testing_data_obj", "wb")

pickle.dump(training_data, training_data_obj)
pickle.dump(testing_data, testing_data_obj)
pickle.dump(validation_data, validation_data_obj)

training_data_obj.close()
validation_data_obj.close()
testing_data_obj.close()

training_length = training_data.__len__()
validation_length = validation_data.__len__()
testing_length = testing_data.__len__()
dataset_length = training_length + validation_length + testing_length

# Length of dataset: 2308
print("Length of dataset: " + str(dataset_length))
print("Length of training dataset: " + str(training_length))
print("Length of validation dataset: " + str(validation_length))
print("Length of testing dataset: " + str(testing_length))

print("Training-Valdation-Testing Split = {}-{}-{}".format(round(training_length/dataset_length*100, 2),
                                                           round(validation_length/dataset_length*100, 2),
                                                           round(testing_length/dataset_length*100, 2)))
