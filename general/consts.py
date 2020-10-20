class DirectoryConsts:
    small_dataset_root = "small_data/personal_data/"
    small_results_directory = "small_data/results/"
    small_model_save_directory = "small_data/models/"
    dataset_root = 'full_data/data_objects/'
    results_directory = "full_data/results/"
    model_save_directory = "full_data/models/"


class FileNames:
    training_data_obj = "training_data_obj"
    validation_data_obj = "validation_data_obj"
    testing_data_obj = "testing_data_obj"


class TrainingConsts:
    train = False
    name = "CNN"
    batch_size = 256
    epoch_count = 150
    shuffle_flag = False
    learning_rate = 0.0005
    checkpoint_frequency = 20
    momentum = 0.9
    scheduler = False
    save_flag = True


class TestingConsts:
    test = True
    test_model_name = 'CNN_860_256_0.0005'
